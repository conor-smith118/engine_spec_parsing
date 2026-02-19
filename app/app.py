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
from typing import Any

import pandas as pd
from dash import Dash, dcc, html, dash_table, callback, Input, Output, State, no_update
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (hardcoded for this workspace; table created on first ingest)
# ---------------------------------------------------------------------------
DATABRICKS_HOST = "https://adb-878896594214094.14.azuredatabricks.net"
HTTP_PATH = "/sql/1.0/warehouses/2e35be694d7a3467"
RESULTS_TABLE = "conor_smith.engine_specs_parse.parsed_engine_data"
UPLOAD_VOLUME = "conor_smith.engine_specs_parse.app_storage"
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "kie-c2e65325-endpoint")

# ---------------------------------------------------------------------------
# SQL connection (cookbook pattern)
# ---------------------------------------------------------------------------
_cfg = None


def get_config():
    global _cfg
    if _cfg is None:
        _cfg = Config(host=os.environ.get("DATABRICKS_HOST", DATABRICKS_HOST))
    return _cfg


def _sql_credentials_provider():
    """Return a token string for the SQL connector (it expects a callable that returns token, not a dict)."""
    cfg = get_config()
    # Prefer explicit token on config (e.g. from DATABRICKS_TOKEN in Apps)
    if hasattr(cfg, "token") and cfg.token:
        return cfg.token
    auth = cfg.authenticate()
    if isinstance(auth, dict):
        return auth.get("token") or auth.get("access_token") or auth.get("bearer")
    if hasattr(auth, "token"):
        return getattr(auth, "token", auth)
    if isinstance(auth, (list, tuple)) and auth:
        return auth[0]
    return auth


def get_connection(http_path: str):
    """Non-cached connection (for when http_path can change)."""
    cfg = get_config()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=_sql_credentials_provider,
    )


def get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient(config=get_config())


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
# ---------------------------------------------------------------------------
def extract_text_from_parsed(parsed: dict) -> str:
    try:
        doc = parsed.get("document") or {}
        elements = doc.get("elements") or []
        parts = [el.get("content", "").strip() for el in elements if isinstance(el.get("content"), str) and el.get("content", "").strip()]
        return "\n\n".join(parts)
    except Exception as e:
        logger.exception("extract_text_from_parsed: %s", e)
        return ""


def invoke_extraction_agent(text: str, endpoint: str) -> dict:
    w = get_workspace_client()
    last_error = None
    for payload in ({"prompt": text}, {"input": text}, {"inputs": [text]}):
        try:
            response = w.serving_endpoints.query(name=endpoint, **payload)
            last_error = None
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise last_error or RuntimeError("Agent invocation failed")
    out = response
    if hasattr(out, "predictions") and out.predictions:
        raw = out.predictions[0] if isinstance(out.predictions[0], str) else str(out.predictions[0])
    elif hasattr(out, "as_dict"):
        d = out.as_dict()
        raw = d.get("predictions", [d])[0] if isinstance(d.get("predictions"), list) else json.dumps(d)
        if isinstance(raw, dict):
            return raw
        raw = str(raw)
    else:
        raw = str(out)
    try:
        if isinstance(raw, str) and raw.strip().startswith("{"):
            return json.loads(raw)
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        if start >= 0:
            end = raw.rfind("}") + 1
            if end > start:
                return json.loads(raw[start:end])
        raise


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
    full_path = f"{volume_path.rstrip('/')}/{file_name}"
    path_esc = full_path.replace("'", "''")
    sql_query = f"""
    SELECT ai_parse_document(content) AS parsed
    FROM READ_FILES('{path_esc}', format => 'binaryFile')
    LIMIT 1
    """
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
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
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        dcc.Link(html.Button("Ingest", style={"marginRight": "8px"}), href="/"),
        dcc.Link(html.Button("Explore", style={"marginRight": "8px"}), href="/explore"),
    ], style={"padding": "12px", "borderBottom": "1px solid #ccc"}),
    html.Div(id="page-content"),
])


def ingest_layout():
    volume_path_default = f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    return html.Div([
        html.H1("Engine spec ingest"),
        html.P("Upload a PDF to parse, extract engine data, and append to the results table."),
        html.Div([
            html.Label("SQL warehouse HTTP path"),
            dcc.Input(id="ingest-http-path", type="text", value=HTTP_PATH, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        html.Div([
            html.Label("Results table (catalog.schema.table)"),
            dcc.Input(id="ingest-table-name", type="text", value=RESULTS_TABLE, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        html.Div([
            html.Label("Volume path"),
            dcc.Input(id="ingest-volume-path", type="text", value=volume_path_default, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        html.Div([
            html.Label("Agent endpoint name"),
            dcc.Input(id="ingest-agent-endpoint", type="text", placeholder=AGENT_ENDPOINT, value=AGENT_ENDPOINT, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        dcc.Upload(
            id="ingest-upload",
            children=html.Button("Select PDF", style={"marginBottom": "8px"}),
            accept=".pdf",
            multiple=False,
        ),
        html.Div(id="ingest-upload-filename", style={"marginBottom": "8px"}),
        html.Button("Parse & ingest", id="ingest-run-btn", n_clicks=0, style={"marginBottom": "8px"}),
        html.Div(id="ingest-error", style={"color": "red", "marginBottom": "8px"}),
        html.Hr(),
        html.Div([
            html.P("Edit rows and click Save to finalize."),
            dash_table.DataTable(
                id="ingest-table",
                columns=[{"name": c, "id": c} for c in RESULTS_COLUMNS],
                data=[],
                editable=True,
                page_action="none",
                style_table={"overflowX": "auto"},
            ),
            html.Button("Save", id="ingest-save-btn", n_clicks=0, style={"marginTop": "8px"}),
        ], id="ingest-table-container"),
        dcc.Store(id="ingest-upload-store", data={"contents": None, "filename": None}),
    ], style={"padding": "24px"})


def explore_layout():
    return html.Div([
        html.H1("Explore engine specs"),
        html.P("Filter and load data, then view, edit, and save."),
        html.Div([
            html.Label("SQL warehouse HTTP path"),
            dcc.Input(id="explore-http-path", type="text", value=HTTP_PATH, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        html.Div([
            html.Label("Results table (catalog.schema.table)"),
            dcc.Input(id="explore-table-name", type="text", value=RESULTS_TABLE, style={"width": "100%", "marginBottom": "8px"}),
        ]),
        html.Div([
            html.Div([html.Label("Manufacturer"), dcc.Input(id="filter-manufacturer", placeholder="e.g. DEUTZ", style={"width": "100%"})], style={"display": "inline-block", "marginRight": "16px", "width": "200px"}),
            html.Div([html.Label("Ingest date"), dcc.Input(id="filter-ingest-date", placeholder="YYYY-MM-DD", style={"width": "100%"})], style={"display": "inline-block", "marginRight": "16px", "width": "120px"}),
            html.Div([html.Label("Cylinder count"), dcc.Input(id="filter-cylinder-count", placeholder="e.g. 4", style={"width": "100%"})], style={"display": "inline-block", "marginRight": "16px", "width": "100px"}),
            html.Div([html.Label("Vermeer product"), dcc.Input(id="filter-vermeer-product", placeholder="Filter by product", style={"width": "100%"})], style={"display": "inline-block", "width": "180px"}),
        ], style={"marginBottom": "12px"}),
        html.Button("Load data", id="explore-load-btn", n_clicks=0, style={"marginBottom": "8px"}),
        html.Div(id="explore-error", style={"color": "red", "marginBottom": "8px"}),
        html.Hr(),
        html.Div([
            html.P("Edit rows and click Save to update the table."),
            dash_table.DataTable(
                id="explore-table",
                columns=[{"name": c, "id": c} for c in RESULTS_COLUMNS],
                data=[],
                editable=True,
                page_action="none",
                style_table={"overflowX": "auto"},
            ),
            html.Button("Save", id="explore-save-btn", n_clicks=0, style={"marginTop": "8px"}),
        ], id="explore-table-container"),
    ], style={"padding": "24px"})


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
# Ingest: Parse & ingest button
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-table", "data"),
    Output("ingest-table", "columns"),
    Output("ingest-error", "children"),
    Input("ingest-run-btn", "n_clicks"),
    State("ingest-upload-store", "data"),
    State("ingest-http-path", "value"),
    State("ingest-table-name", "value"),
    State("ingest-volume-path", "value"),
    State("ingest-agent-endpoint", "value"),
    prevent_initial_call=True,
)
def run_ingest(n_clicks, upload_data, http_path, table_name, volume_path, agent_endpoint):
    if not n_clicks or not upload_data or not upload_data.get("contents") or not upload_data.get("filename"):
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Select a PDF and click Parse & ingest."
    http_path = (http_path or "").strip() or HTTP_PATH
    table_name = (table_name or "").strip() or RESULTS_TABLE
    volume_path = (volume_path or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/Volumes/"):
        parts = UPLOAD_VOLUME.split(".")
        volume_path = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    agent_endpoint = (agent_endpoint or "").strip() or AGENT_ENDPOINT
    try:
        content_str = upload_data["contents"]
        if "," in content_str:
            content_str = content_str.split(",")[1]
        file_bytes = base64.b64decode(content_str)
        filename = upload_data["filename"] or "document.pdf"
        w = get_workspace_client()
        w.files.upload(f"{volume_path.rstrip('/')}/{filename}", io.BytesIO(file_bytes), overwrite=True)
        conn = get_connection(http_path)
        ensure_results_table(conn, table_name)
        parsed = run_parse_document_sql(conn, volume_path, filename)
        if not parsed:
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "No parsed document returned from ai_parse_document."
        text = extract_text_from_parsed(parsed)
        if not text.strip():
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Parsed document had no text content."
        agent_out = invoke_extraction_agent(text, agent_endpoint)
        ingest_id = str(uuid.uuid4())
        today = date.today().isoformat()
        exploded = explode_agent_output(agent_out, today, filename, ingest_id)
        insert_rows(conn, table_name, exploded)
        where = f"ingest_id = '{ingest_id.replace(chr(39), chr(39)+chr(39))}'"
        df = read_table(conn, table_name, where)
        data = df.astype(str).replace("nan", "").to_dict("records")
        cols = [{"name": c, "id": c} for c in RESULTS_COLUMNS]
        return data, cols, ""
    except Exception as e:
        logger.exception("run_ingest: %s", e)
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e)


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
