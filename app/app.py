"""
Engine Spec Parsing – Databricks Dash app.

Page 1 (Ingest): Upload PDF → ai_parse_document → Information Extraction Agent
  → explode JSON → append to results table → editable table → Save.
Page 2 (Explore): Filters (Manufacturer, Ingest Date, Cylinder count, Vermeer Product)
  → load into editable table → view/update/save rows.

Cookbook: https://apps-cookbook.dev/docs/category/dash
"""
import base64
import io
import json
import logging
import os
import uuid
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import pandas as pd
import diskcache
from dash import (
    Dash,
    DiskcacheManager,
    dcc,
    html,
    dash_table,
    callback,
    Input,
    Output,
    State,
    no_update,
)
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# Background callbacks (step progress during ingest)
_cache = diskcache.Cache(os.environ.get("DASH_CACHE_DIR", "/tmp/dash_cache"))
_background_callback_manager = DiskcacheManager(_cache)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress per-request logs (e.g. background callback polling every ~2s)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config (hardcoded for this workspace; table created on first ingest)
# ---------------------------------------------------------------------------
DATABRICKS_HOST = "https://adb-878896594214094.14.azuredatabricks.net"
HTTP_PATH = "/sql/1.0/warehouses/2e35be694d7a3467"
RESULTS_TABLE = "conor_smith.engine_specs_parse.parsed_engine_data"
UPLOAD_VOLUME = "conor_smith.engine_specs_parse.app_storage"
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "kie-c2e65325-endpoint")

# ---------------------------------------------------------------------------
# SQL connection (cookbook: tables_read / tables_edit)
# https://apps-cookbook.dev/docs/dash/tables/tables_read
# https://apps-cookbook.dev/docs/dash/tables/tables_edit
# ---------------------------------------------------------------------------
cfg = Config(host=os.environ.get("DATABRICKS_HOST", DATABRICKS_HOST))


@lru_cache(maxsize=1)
def get_connection(http_path: str):
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


def get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient(config=cfg)


def _format_sql_val(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NULL"
    if isinstance(x, str):
        return "'" + x.replace("'", "''") + "'"
    if isinstance(x, (date, datetime, pd.Timestamp)):
        return f"'{x}'"
    return str(x)


# ---------------------------------------------------------------------------
# Table schema and helpers
# ---------------------------------------------------------------------------
RESULTS_COLUMNS = [
    "id", "company", "product_series", "engine_type",
    "power_rating_continuous_operations", "number_of_cylinders",
    "specific_fuel_consumption", "ingest_date", "vermeer_product",
    "source_file", "ingest_id",
]


def ensure_results_table(conn, table_name: str) -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id STRING NOT NULL,
        company STRING,
        product_series STRING,
        engine_type STRING,
        power_rating_continuous_operations STRING,
        number_of_cylinders INT,
        specific_fuel_consumption STRING,
        ingest_date DATE,
        vermeer_product STRING,
        source_file STRING,
        ingest_id STRING
    )
    USING DELTA
    """
    with conn.cursor() as cursor:
        cursor.execute(create_sql)


def read_table(conn, table_name: str, where_clause: str = "") -> pd.DataFrame:
    q = f"SELECT * FROM {table_name}"
    if where_clause:
        q += " WHERE " + where_clause
    q += " ORDER BY ingest_date DESC, id"
    with conn.cursor() as cursor:
        cursor.execute(q)
        return cursor.fetchall_arrow().to_pandas()


def update_rows_by_id(conn, table_name: str, df: pd.DataFrame, id_col: str = "id") -> None:
    if df.empty:
        return
    col_names = [c for c in df.columns if c != id_col]
    for _, row in df.iterrows():
        row_id = row[id_col]
        set_parts = ", ".join(f"{c} = {_format_sql_val(row[c])}" for c in col_names)
        sql_update = f"UPDATE {table_name} SET {set_parts} WHERE id = {_format_sql_val(str(row_id))}"
        with conn.cursor() as cursor:
            cursor.execute(sql_update)


def insert_rows(conn, table_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ", ".join(df.columns)
    rows = []
    for _, row in df.iterrows():
        vals = ", ".join(_format_sql_val(row[c]) for c in df.columns)
        rows.append(f"({vals})")
    sql_insert = f"INSERT INTO {table_name} ({cols}) VALUES {', '.join(rows)}"
    with conn.cursor() as cursor:
        cursor.execute(sql_insert)


# ---------------------------------------------------------------------------
# PDF parse + Agent + Explode
# Logic mirrors the working Spark SQL: metadata.version selects pages vs elements,
# then we concat element content with \n\n. No Spark - SQL only (READ_FILES + ai_parse_document).
# ---------------------------------------------------------------------------
def extract_text_from_parsed(parsed: dict) -> str:
    """Extract full text from ai_parse_document result. Matches Spark SQL: with_raw / concatenated logic."""
    try:
        if not isinstance(parsed, dict):
            return ""
        # Same as SQL: error_status present -> treat as error, no text
        error_status = parsed.get("error_status")
        if error_status is not None and str(error_status).strip():
            logger.warning("extract_text_from_parsed: error_status=%s", error_status)
            return ""
        doc = parsed.get("document") or {}
        metadata = parsed.get("metadata") or {}
        version = (metadata.get("version") or "").strip()
        # Same as SQL: version '1.0' -> document.pages, else document.elements
        if version == "1.0":
            elements = doc.get("pages") or []
        else:
            elements = doc.get("elements") or []
        parts = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            content = el.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        return "\n\n".join(parts)
    except Exception as e:
        logger.exception("extract_text_from_parsed: %s", e)
        return ""


def invoke_extraction_agent(text: str, endpoint: str) -> dict:
    """
    Call the extraction agent (same pattern as ai_query in SQL: result + errorMessage).
    No Spark: uses serving_endpoints.query. Returns the structured extraction dict
    (company, product_series, engines) or raises if errorMessage is set.
    """
    w = get_workspace_client()
    last_error = None
    for payload in (
        {"messages": [{"role": "user", "content": text}]},
        {"prompt": text},
        {"input": text},
        {"inputs": [text]},
    ):
        try:
            response = w.serving_endpoints.query(name=endpoint, **payload)
            last_error = None
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise last_error or RuntimeError("Agent invocation failed")

    # Normalize to one response object (like ai_query returns one row with response.result / response.errorMessage)
    out = response
    if hasattr(out, "predictions") and out.predictions:
        raw = out.predictions[0]
    elif hasattr(out, "as_dict"):
        d = out.as_dict()
        preds = d.get("predictions") or [d]
        raw = preds[0] if preds else {}
    else:
        raw = out

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}") + 1
            raw = json.loads(raw[start:end]) if end > start else {}

    if not isinstance(raw, dict):
        return {}

    # ai_query pattern: response.errorMessage -> fail; response.result -> structured data
    error_message = raw.get("errorMessage") or raw.get("error_message")
    if error_message and str(error_message).strip():
        logger.warning("invoke_extraction_agent: errorMessage=%s", error_message)
        raise RuntimeError(f"Extraction failed: {error_message}")

    # Use result as the extraction payload (ai_query returns response.result)
    contract_data = raw.get("result") or raw
    if not isinstance(contract_data, dict):
        return {}
    return contract_data


def explode_agent_output(agent_output: dict, ingest_date: str, source_file: str, ingest_id: str) -> pd.DataFrame:
    company = agent_output.get("company") or ""
    product_series = agent_output.get("product_series") or ""
    engines = agent_output.get("engines") or []
    rows = []
    for e in engines:
        rows.append({
            "id": str(uuid.uuid4()),
            "company": company,
            "product_series": product_series,
            "engine_type": (e.get("engine_type") or ""),
            "power_rating_continuous_operations": (e.get("power_rating_continuous_operations") or ""),
            "number_of_cylinders": e.get("number_of_cylinders"),
            "specific_fuel_consumption": (e.get("specific_fuel_consumption") or ""),
            "ingest_date": ingest_date,
            "vermeer_product": "",
            "source_file": source_file,
            "ingest_id": ingest_id,
        })
    return pd.DataFrame(rows)


def run_parse_document_sql(conn, volume_path: str, file_name: str) -> dict | None:
    """
    Parse one file with ai_parse_document via warehouse SQL only (no Spark).
    Same semantic as the Spark flow: READ_FILES -> ai_parse_document(content).
    Path must be a single file path; READ_FILES returns (path, content, ...).
    """
    full_path = f"{volume_path.rstrip('/')}/{file_name}"
    logger.info("run_parse_document_sql: path=%s", full_path)
    path_esc = full_path.replace("'", "''")
    sql_query = f"""
    SELECT ai_parse_document(content) AS parsed
    FROM READ_FILES('{path_esc}', format => 'binaryFile')
    LIMIT 1
    """
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        logger.info("run_parse_document_sql: query executed, fetching results")
        tbl = cursor.fetchall_arrow()
    if tbl is None or tbl.num_rows == 0:
        return None
    df = tbl.to_pandas()
    val = df.iloc[0]["parsed"]
    if hasattr(val, "as_py"):
        val = val.as_py()
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return None


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    background_callback_manager=_background_callback_manager,
)

# Shared styles
NAV_STYLE = {
    "padding": "12px 24px",
    "borderBottom": "1px solid #e2e8f0",
    "backgroundColor": "#0f172a",
    "display": "flex",
    "gap": "8px",
    "alignItems": "center",
}
NAV_LINK_STYLE = {
    "color": "#f8fafc",
    "textDecoration": "none",
    "padding": "8px 16px",
    "borderRadius": "8px",
    "fontWeight": "500",
    "backgroundColor": "transparent",
    "border": "1px solid transparent",
}
PAGE_STYLE = {
    "maxWidth": "1200px",
    "margin": "0 auto",
    "padding": "32px 24px",
    "fontFamily": "'Segoe UI', system-ui, sans-serif",
    "backgroundColor": "#f8fafc",
    "minHeight": "100vh",
}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "padding": "24px",
    "marginBottom": "24px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    "border": "1px solid #e2e8f0",
}
LABEL_STYLE = {"display": "block", "fontWeight": "600", "marginBottom": "6px", "color": "#334155"}
INPUT_STYLE = {
    "width": "100%",
    "padding": "10px 12px",
    "borderRadius": "8px",
    "border": "1px solid #cbd5e1",
    "fontSize": "14px",
    "marginBottom": "16px",
}
BTN_PRIMARY = {
    "padding": "10px 20px",
    "borderRadius": "8px",
    "border": "none",
    "fontWeight": "600",
    "fontSize": "14px",
    "cursor": "pointer",
    "backgroundColor": "#2563eb",
    "color": "#ffffff",
}
BTN_SECONDARY = {**BTN_PRIMARY, "backgroundColor": "#64748b", "color": "#ffffff"}
BTN_SUCCESS = {**BTN_PRIMARY, "backgroundColor": "#059669"}

STEP_NAMES = ["Parsing Document", "Extracting Information", "Writing Data", "Done"]


def step_tracker(current: str):
    """Build step tracker UI: four steps with current highlighted."""
    idx = STEP_NAMES.index(current) if current in STEP_NAMES else -1
    steps = []
    for i, name in enumerate(STEP_NAMES):
        is_active = current == name
        is_done = idx > i
        step_style = {
            "flex": "1",
            "textAlign": "center",
            "padding": "12px 8px",
            "borderRadius": "8px",
            "fontWeight": "600" if is_active else "500",
            "fontSize": "13px",
            "backgroundColor": "#2563eb" if is_active else "#e2e8f0" if is_done else "#f1f5f9",
            "color": "#ffffff" if is_active else "#64748b" if is_done else "#94a3b8",
            "border": "2px solid #2563eb" if is_active else "2px solid transparent",
        }
        steps.append(html.Div(name, style=step_style))
    return html.Div(
        steps,
        style={
            "display": "flex",
            "gap": "8px",
            "marginBottom": "24px",
            "flexWrap": "wrap",
        },
    )


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Nav(
            [
                html.Div("Engine Spec Parsing", style={"color": "#f8fafc", "fontWeight": "700", "fontSize": "18px", "marginRight": "24px"}),
                dcc.Link(html.Span("Ingest", style=NAV_LINK_STYLE), href="/", id="nav-ingest"),
                dcc.Link(html.Span("Explore", style=NAV_LINK_STYLE), href="/explore", id="nav-explore"),
            ],
            style=NAV_STYLE,
        ),
        html.Div(id="page-content", style={"backgroundColor": "#f8fafc"}),
    ],
    style={"margin": 0, "padding": 0},
)


def ingest_layout():
    volume_path_default = f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Ingest engine specs", style={"margin": "0 0 8px 0", "color": "#0f172a", "fontSize": "28px"}),
                    html.P("Upload a PDF to parse, extract engine data, and append to the results table.", style={"margin": 0, "color": "#64748b"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Configuration", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.Label("SQL warehouse HTTP path", style=LABEL_STYLE),
                    dcc.Input(id="ingest-http-path", type="text", value=HTTP_PATH, style=INPUT_STYLE),
                    html.Label("Results table (catalog.schema.table)", style=LABEL_STYLE),
                    dcc.Input(id="ingest-table-name", type="text", value=RESULTS_TABLE, style=INPUT_STYLE),
                    html.Label("Volume path", style=LABEL_STYLE),
                    dcc.Input(id="ingest-volume-path", type="text", value=volume_path_default, style=INPUT_STYLE),
                    html.Label("Agent endpoint name", style=LABEL_STYLE),
                    dcc.Input(id="ingest-agent-endpoint", type="text", value=AGENT_ENDPOINT, style=INPUT_STYLE),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Upload & run", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    dcc.Upload(
                        id="ingest-upload",
                        children=html.Div(["Select PDF or drag here"], style={"padding": "20px", "border": "2px dashed #cbd5e1", "borderRadius": "8px", "textAlign": "center", "color": "#64748b", "cursor": "pointer"}),
                        accept=".pdf",
                        multiple=False,
                    ),
                    html.Div(id="ingest-upload-filename", style={"marginTop": "8px", "fontSize": "14px", "color": "#475569"}),
                    html.Div(
                        [
                            html.Button("Parse & ingest", id="ingest-run-btn", n_clicks=0, style=BTN_SUCCESS),
                        ],
                        style={"marginTop": "16px"},
                    ),
                    html.Div(id="ingest-progress", style={"marginTop": "20px"}),
                    html.Div(id="ingest-error", style={"color": "#dc2626", "marginTop": "12px", "fontSize": "14px"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Extracted data", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.P("Edit rows and click Save to finalize.", style={"margin": "0 0 12px 0", "color": "#64748b"}),
                    dash_table.DataTable(
                        id="ingest-table",
                        columns=[{"name": c, "id": c} for c in RESULTS_COLUMNS],
                        data=[],
                        editable=True,
                        page_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "10px", "fontSize": "13px"},
                        style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    ),
                    html.Button("Save changes", id="ingest-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
                ],
                style=CARD_STYLE,
                id="ingest-table-container",
            ),
            dcc.Store(id="ingest-upload-store", data={"contents": None, "filename": None}),
        ],
        style=PAGE_STYLE,
    )


def explore_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Explore engine specs", style={"margin": "0 0 8px 0", "color": "#0f172a", "fontSize": "28px"}),
                    html.P("Filter and load data, then view, edit, and save.", style={"margin": 0, "color": "#64748b"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Configuration", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.Label("SQL warehouse HTTP path", style=LABEL_STYLE),
                    dcc.Input(id="explore-http-path", type="text", value=HTTP_PATH, style=INPUT_STYLE),
                    html.Label("Results table (catalog.schema.table)", style=LABEL_STYLE),
                    dcc.Input(id="explore-table-name", type="text", value=RESULTS_TABLE, style=INPUT_STYLE),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Filters", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.Div(
                        [
                            html.Div([html.Label("Manufacturer", style=LABEL_STYLE), dcc.Input(id="filter-manufacturer", placeholder="e.g. DEUTZ", style=INPUT_STYLE)], style={"display": "inline-block", "marginRight": "16px", "width": "200px", "verticalAlign": "top"}),
                            html.Div([html.Label("Ingest date", style=LABEL_STYLE), dcc.Input(id="filter-ingest-date", placeholder="YYYY-MM-DD", style=INPUT_STYLE)], style={"display": "inline-block", "marginRight": "16px", "width": "140px", "verticalAlign": "top"}),
                            html.Div([html.Label("Cylinder count", style=LABEL_STYLE), dcc.Input(id="filter-cylinder-count", placeholder="e.g. 4", style=INPUT_STYLE)], style={"display": "inline-block", "marginRight": "16px", "width": "120px", "verticalAlign": "top"}),
                            html.Div([html.Label("Vermeer product", style=LABEL_STYLE), dcc.Input(id="filter-vermeer-product", placeholder="Filter by product", style=INPUT_STYLE)], style={"display": "inline-block", "width": "200px", "verticalAlign": "top"}),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Button("Load data", id="explore-load-btn", n_clicks=0, style=BTN_SUCCESS),
                    html.Div(id="explore-error", style={"color": "#dc2626", "marginTop": "12px", "fontSize": "14px"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Data", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.P("Edit rows and click Save to update the table.", style={"margin": "0 0 12px 0", "color": "#64748b"}),
                    dash_table.DataTable(
                        id="explore-table",
                        columns=[{"name": c, "id": c} for c in RESULTS_COLUMNS],
                        data=[],
                        editable=True,
                        page_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "10px", "fontSize": "13px"},
                        style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    ),
                    html.Button("Save changes", id="explore-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
                ],
                style=CARD_STYLE,
                id="explore-table-container",
            ),
        ],
        style=PAGE_STYLE,
    )


@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/explore":
        return explore_layout()
    return ingest_layout()


# ---------------------------------------------------------------------------
# Ingest: store upload
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-upload-store", "data"),
    Output("ingest-upload-filename", "children"),
    Input("ingest-upload", "contents"),
    State("ingest-upload", "filename"),
)
def store_upload(contents, filename):
    if contents is None:
        return no_update, ""
    return {"contents": contents, "filename": filename}, html.Span(f"Selected: {filename or '?'}")


# ---------------------------------------------------------------------------
# Ingest: Parse & ingest button (background callback with step progress)
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("ingest-table", "data"),
        Output("ingest-table", "columns"),
        Output("ingest-error", "children"),
        Output("ingest-progress", "children"),
    ),
    inputs=Input("ingest-run-btn", "n_clicks"),
    state=[
        State("ingest-upload-store", "data"),
        State("ingest-http-path", "value"),
        State("ingest-table-name", "value"),
        State("ingest-volume-path", "value"),
        State("ingest-agent-endpoint", "value"),
    ],
    background=True,
    progress=[Output("ingest-progress", "children")],
    prevent_initial_call=True,
)
def run_ingest(set_progress, n_clicks, upload_data, http_path, table_name, volume_path, agent_endpoint):
    empty_result = [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Select a PDF and click Parse & ingest.", step_tracker("")
    if not n_clicks or not upload_data or not upload_data.get("contents") or not upload_data.get("filename"):
        return empty_result
    http_path = (http_path or "").strip() or HTTP_PATH
    table_name = (table_name or "").strip() or RESULTS_TABLE
    volume_path = (volume_path or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/Volumes/"):
        parts = UPLOAD_VOLUME.split(".")
        volume_path = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    agent_endpoint = (agent_endpoint or "").strip() or AGENT_ENDPOINT
    try:
        set_progress(step_tracker("Parsing Document"))
        logger.info("Ingest: decoding upload and uploading file to volume")
        content_str = upload_data["contents"]
        if "," in content_str:
            content_str = content_str.split(",")[1]
        file_bytes = base64.b64decode(content_str)
        filename = upload_data["filename"] or "document.pdf"
        w = get_workspace_client()
        upload_path = f"{volume_path.rstrip('/')}/{filename}"
        w.files.upload(upload_path, io.BytesIO(file_bytes), overwrite=True)
        logger.info("Ingest: file uploaded to %s", upload_path)

        logger.info("Ingest: connecting to SQL warehouse (http_path=%s)", http_path[:50] + "..." if len(http_path) > 50 else http_path)
        conn = get_connection(http_path)
        logger.info("Ingest: ensuring results table exists")
        ensure_results_table(conn, table_name)

        logger.info("Ingest: running READ_FILES + ai_parse_document (this may take a minute)")
        parsed = run_parse_document_sql(conn, volume_path, filename)
        if not parsed:
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "No parsed document returned from ai_parse_document.", step_tracker("")
        text = extract_text_from_parsed(parsed)
        if not text.strip():
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Parsed document had no text content.", step_tracker("")

        set_progress(step_tracker("Extracting Information"))
        agent_out = invoke_extraction_agent(text, agent_endpoint)
        ingest_id = str(uuid.uuid4())
        today = date.today().isoformat()
        exploded = explode_agent_output(agent_out, today, filename, ingest_id)

        set_progress(step_tracker("Writing Data"))
        insert_rows(conn, table_name, exploded)
        where = f"ingest_id = '{ingest_id.replace(chr(39), chr(39)+chr(39))}'"
        df = read_table(conn, table_name, where)
        data = df.astype(str).replace("nan", "").to_dict("records")
        cols = [{"name": c, "id": c} for c in RESULTS_COLUMNS]

        set_progress(step_tracker("Done"))
        return data, cols, "", step_tracker("Done")
    except Exception as e:
        logger.exception("run_ingest: %s", e)
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e), step_tracker("")


# ---------------------------------------------------------------------------
# Ingest: Save button
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-error", "children", allow_duplicate=True),
    Input("ingest-save-btn", "n_clicks"),
    State("ingest-table", "data"),
    State("ingest-table", "columns"),
    State("ingest-http-path", "value"),
    State("ingest-table-name", "value"),
    prevent_initial_call=True,
)
def save_ingest(n_clicks, data, columns, http_path, table_name):
    if not n_clicks or not data or not columns:
        return "No data to save."
    http_path = (http_path or "").strip() or HTTP_PATH
    table_name = (table_name or "").strip() or RESULTS_TABLE
    try:
        col_names = [c["id"] for c in columns]
        df = pd.DataFrame(data, columns=col_names)
        conn = get_connection(http_path)
        update_rows_by_id(conn, table_name, df)
        return "Saved successfully."
    except Exception as e:
        logger.exception("save_ingest: %s", e)
        return f"Error saving: {e}"


# ---------------------------------------------------------------------------
# Explore: Load data
# ---------------------------------------------------------------------------
@callback(
    Output("explore-table", "data"),
    Output("explore-table", "columns"),
    Output("explore-error", "children"),
    Input("explore-load-btn", "n_clicks"),
    State("explore-http-path", "value"),
    State("explore-table-name", "value"),
    State("filter-manufacturer", "value"),
    State("filter-ingest-date", "value"),
    State("filter-cylinder-count", "value"),
    State("filter-vermeer-product", "value"),
    prevent_initial_call=True,
)
def load_explore(n_clicks, http_path, table_name, f_man, f_date, f_cyl, f_vermeer):
    if not n_clicks:
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], ""
    http_path = (http_path or "").strip() or HTTP_PATH
    table_name = (table_name or "").strip() or RESULTS_TABLE
    parts = []
    if f_man and str(f_man).strip():
        parts.append(f"company = '{str(f_man).strip().replace(chr(39), chr(39)+chr(39))}'")
    if f_date and str(f_date).strip():
        parts.append(f"ingest_date = '{str(f_date).strip()}'")
    if f_cyl and str(f_cyl).strip():
        parts.append(f"number_of_cylinders = {str(f_cyl).strip()}")
    if f_vermeer and str(f_vermeer).strip():
        parts.append(f"vermeer_product = '{str(f_vermeer).strip().replace(chr(39), chr(39)+chr(39))}'")
    where = " AND ".join(parts) if parts else "1=1"
    try:
        conn = get_connection(http_path)
        df = read_table(conn, table_name, where)
        data = df.astype(str).replace("nan", "").to_dict("records")
        cols = [{"name": c, "id": c} for c in RESULTS_COLUMNS]
        return data, cols, ""
    except Exception as e:
        logger.exception("load_explore: %s", e)
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e)


# ---------------------------------------------------------------------------
# Explore: Save button
# ---------------------------------------------------------------------------
@callback(
    Output("explore-error", "children", allow_duplicate=True),
    Input("explore-save-btn", "n_clicks"),
    State("explore-table", "data"),
    State("explore-table", "columns"),
    State("explore-http-path", "value"),
    State("explore-table-name", "value"),
    prevent_initial_call=True,
)
def save_explore(n_clicks, data, columns, http_path, table_name):
    if not n_clicks or not data or not columns:
        return "No data to save."
    http_path = (http_path or "").strip() or HTTP_PATH
    table_name = (table_name or "").strip() or RESULTS_TABLE
    try:
        col_names = [c["id"] for c in columns]
        df = pd.DataFrame(data, columns=col_names)
        conn = get_connection(http_path)
        update_rows_by_id(conn, table_name, df)
        return "Saved successfully."
    except Exception as e:
        logger.exception("save_explore: %s", e)
        return f"Error saving: {e}"


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", "8080")))
    app.run(host="0.0.0.0", port=port, debug=False)
