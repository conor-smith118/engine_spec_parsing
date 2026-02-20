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
import time
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
# One-row table for ai_query input (same pattern as newly_parsed_temp -> ai_query in SQL)
EXTRACTION_INPUT_TABLE = "conor_smith.engine_specs_parse._extraction_input"

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


def ensure_extraction_input_table(conn, table_name: str) -> None:
    """Create or replace the one-row table used as input to ai_query (path, text). Same pattern as Spark createOrReplaceTempView / saveAsTable — we need the exact schema."""
    create_sql = f"""
    CREATE OR REPLACE TABLE {table_name} (
        path STRING NOT NULL,
        text STRING
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


def read_distinct_filter_options(conn, table_name: str) -> dict:
    """Fetch distinct company, number_of_cylinders, vermeer_product for Explore filter dropdowns."""
    out = {"company": [], "number_of_cylinders": [], "vermeer_product": []}
    try:
        for col, key in [("company", "company"), ("number_of_cylinders", "number_of_cylinders"), ("vermeer_product", "vermeer_product")]:
            q = f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL ORDER BY {col}"
            with conn.cursor() as cursor:
                cursor.execute(q)
                df = cursor.fetchall_arrow().to_pandas()
            if not df.empty and col in df.columns:
                vals = df[col].dropna().astype(str).str.strip().unique().tolist()
                if col == "number_of_cylinders":
                    vals = sorted([v for v in vals if v], key=lambda x: (int(x) if x.isdigit() else 999))
                else:
                    vals = sorted([v for v in vals if v])
                out[key] = vals
    except Exception as e:
        logger.warning("read_distinct_filter_options: %s", e)
    return out


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


def invoke_extraction_agent_sql(conn, path: str, text: str, endpoint: str) -> dict:
    """
    Extraction via SQL ai_query (same pattern as your working flow). No Spark, no REST.
    Uses a one-row table (path, text), runs ai_query(endpoint, text, failOnError => false),
    then reads response.result / response.errorMessage.
    """
    table = EXTRACTION_INPUT_TABLE
    ensure_extraction_input_table(conn, table)
    endpoint_esc = endpoint.replace("'", "''")

    with conn.cursor() as cursor:
        cursor.execute(f"DELETE FROM {table}")
        cursor.execute(
            f"INSERT INTO {table} (path, text) VALUES (?, ?)",
            parameters=[path, text],
        )

    # Same as your pattern: FROM table -> ai_query(endpoint, text, failOnError => false) AS response
    query = f"""
    SELECT
        path,
        text AS input,
        ai_query('{endpoint_esc}', text, failOnError => false) AS response
    FROM {table}
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
        tbl = cursor.fetchall_arrow()
    if tbl is None or tbl.num_rows == 0:
        raise RuntimeError("ai_query returned no row")

    row = tbl.to_pandas().iloc[0]
    response = row["response"]
    if hasattr(response, "as_py"):
        response = response.as_py()
    if response is None:
        logger.error("invoke_extraction_agent_sql: ai_query response was null")
        raise RuntimeError("ai_query response was null (endpoint error)")

    if isinstance(response, dict):
        raw = response
    elif isinstance(response, str):
        raw = json.loads(response) if response.strip().startswith("{") else {}
    else:
        raw = {}
    logger.info("invoke_extraction_agent_sql: response keys=%s", list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__)

    error_message = raw.get("errorMessage") or raw.get("error_message") or raw.get("errorStatus")
    if error_message and str(error_message).strip():
        logger.warning("invoke_extraction_agent_sql: errorMessage=%s", error_message)
        raise RuntimeError(f"Extraction failed: {error_message}")

    # ai_query returns struct with result (and optionally errorMessage). Fallback to predictions / candidates.
    contract_data = raw.get("result") or raw.get("predictions") or raw
    if isinstance(contract_data, str) and contract_data.strip().startswith("{"):
        try:
            contract_data = json.loads(contract_data)
        except json.JSONDecodeError:
            contract_data = {}
    if isinstance(contract_data, list) and contract_data:
        contract_data = contract_data[0] if isinstance(contract_data[0], dict) else {}
    if not isinstance(contract_data, dict):
        logger.warning("invoke_extraction_agent_sql: result not a dict, type=%s", type(contract_data).__name__)
        return {}
    logger.info("invoke_extraction_agent_sql: extraction keys=%s, engines count=%s", list(contract_data.keys()), len(contract_data.get("engines") or []))
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
    t0 = time.time()
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        t1 = time.time()
        tbl = cursor.fetchall_arrow()
        t2 = time.time()
    if tbl is None or tbl.num_rows == 0:
        return None
    df = tbl.to_pandas()
    val = df.iloc[0]["parsed"]
    if hasattr(val, "as_py"):
        val = val.as_py()
    t3 = time.time()
    execute_s = t1 - t0
    fetch_s = t2 - t1
    post_s = t3 - t2
    total_s = t3 - t0
    logger.info(
        "run_parse_document_sql: timing execute=%.1fs fetch=%.1fs post_process=%.1fs total=%.1fs",
        execute_s, fetch_s, post_s, total_s,
    )
    logger.info("run_parse_document_sql: got parsed value, type=%s", type(val).__name__)
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

# Pulse animation name/duration (keyframes defined in assets/custom.css)
PULSE_ANIMATION = "pulse-opacity 2s ease-in-out infinite"

STEP_NAMES = ["Parsing Document", "Extracting Information", "Writing Data", "Done"]

# Default app config (Admin page edits this store)
def _default_app_config():
    return {
        "http_path": HTTP_PATH,
        "results_table": RESULTS_TABLE,
        "volume_path": f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}",
        "agent_endpoint": AGENT_ENDPOINT,
    }


def _ingest_save_container(phase: str):
    """phase: 'normal' | 'saving' | 'saved'"""
    if phase == "saving":
        return html.Div([
            html.Button("Saving", id="ingest-save-btn", disabled=True, style={**BTN_PRIMARY, "marginTop": "16px", "animation": PULSE_ANIMATION}),
            html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
        ], id="ingest-save-container", style={"display": "inline-flex", "alignItems": "center"})
    if phase == "saved":
        return html.Div([
            html.Button("Save changes", id="ingest-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
            html.Span(" ✓ Saved!", style={"color": "#059669", "marginLeft": "8px", "fontWeight": "600"}),
        ], id="ingest-save-container", style={"display": "inline-flex", "alignItems": "center"})
    return html.Div([
        html.Button("Save changes", id="ingest-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
        html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
    ], id="ingest-save-container", style={"display": "inline-flex", "alignItems": "center"})


def _explore_save_container(phase: str):
    if phase == "saving":
        return html.Div([
            html.Button("Saving", id="explore-save-btn", disabled=True, style={**BTN_PRIMARY, "marginTop": "16px", "animation": PULSE_ANIMATION}),
            html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
        ], id="explore-save-container", style={"display": "inline-flex", "alignItems": "center"})
    if phase == "saved":
        return html.Div([
            html.Button("Save changes", id="explore-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
            html.Span(" ✓ Saved!", style={"color": "#059669", "marginLeft": "8px", "fontWeight": "600"}),
        ], id="explore-save-container", style={"display": "inline-flex", "alignItems": "center"})
    return html.Div([
        html.Button("Save changes", id="explore-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
        html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
    ], id="explore-save-container", style={"display": "inline-flex", "alignItems": "center"})


def _explore_load_container(phase: str):
    """phase: 'normal' | 'loading' | 'loaded'"""
    if phase == "loading":
        return html.Div([
            html.Button("Loading", id="explore-load-btn", disabled=True, style={**BTN_SUCCESS, "animation": PULSE_ANIMATION}),
            html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
        ], id="explore-load-container", style={"display": "inline-flex", "alignItems": "center"})
    if phase == "loaded":
        return html.Div([
            html.Button("Load data", id="explore-load-btn", n_clicks=0, style=BTN_SUCCESS),
            html.Span(" ✓ Loaded!", style={"color": "#059669", "marginLeft": "8px", "fontWeight": "600"}),
        ], id="explore-load-container", style={"display": "inline-flex", "alignItems": "center"})
    return html.Div([
        html.Button("Load data", id="explore-load-btn", n_clicks=0, style=BTN_SUCCESS),
        html.Span(style={"marginLeft": "8px", "fontWeight": "600"}),
    ], id="explore-load-container", style={"display": "inline-flex", "alignItems": "center"})


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
        if is_active and name != "Done":
            step_style["animation"] = PULSE_ANIMATION
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


# Tab style: large, pill-like; active tab is filled (clearly visible and centered)
def _nav_tab_style(is_active: bool):
    base = {
        "padding": "22px 52px",
        "borderRadius": "12px",
        "fontWeight": "600",
        "fontSize": "20px",
        "textDecoration": "none",
        "border": "2px solid transparent",
    }
    if is_active:
        base["backgroundColor"] = "#2563eb"
        base["color"] = "#ffffff"
        base["borderColor"] = "#2563eb"
    else:
        base["backgroundColor"] = "transparent"
        base["color"] = "#e2e8f0"
        base["borderColor"] = "#475569"
    return base


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="app-config", data=_default_app_config()),
        dcc.Store(id="explore-filter-options", data=None),
        html.Div(id="nav-container", style=NAV_STYLE),
        html.Div(id="page-content", style={"backgroundColor": "#f8fafc"}),
    ],
    style={"margin": 0, "padding": 0},
)


def ingest_layout():
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
                    html.Div(
                        [
                            html.Button("Save changes", id="ingest-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
                            html.Span(id="ingest-save-status", style={"marginLeft": "8px", "fontWeight": "600"}),
                        ],
                        id="ingest-save-container",
                        style={"display": "inline-flex", "alignItems": "center"},
                    ),
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
                    html.H3("Filters", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.Div(
                        [
                            html.Div(
                                [html.Label("Manufacturer", style=LABEL_STYLE), dcc.Dropdown(id="filter-manufacturer", options=[], value=None, placeholder="All manufacturers", clearable=True, style={"marginBottom": "16px"})],
                                style={"display": "inline-block", "marginRight": "16px", "width": "220px", "verticalAlign": "top"},
                            ),
                            html.Div(
                                [html.Label("Ingest date", style=LABEL_STYLE), dcc.DatePickerSingle(id="filter-ingest-date", placeholder="Select date", clearable=True, display_format="YYYY-MM-DD")],
                                style={"display": "inline-block", "marginRight": "16px", "width": "180px", "verticalAlign": "top"},
                            ),
                            html.Div(
                                [html.Label("Cylinder count", style=LABEL_STYLE), dcc.Dropdown(id="filter-cylinder-count", options=[], value=None, placeholder="All", clearable=True, style={"marginBottom": "16px"})],
                                style={"display": "inline-block", "marginRight": "16px", "width": "140px", "verticalAlign": "top"},
                            ),
                            html.Div(
                                [html.Label("Vermeer product", style=LABEL_STYLE), dcc.Dropdown(id="filter-vermeer-product", options=[], value=None, placeholder="All products", clearable=True, style={"marginBottom": "16px"})],
                                style={"display": "inline-block", "width": "220px", "verticalAlign": "top"},
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Button("Load data", id="explore-load-btn", n_clicks=0, style=BTN_SUCCESS),
                            html.Span(id="explore-load-status", style={"marginLeft": "8px", "fontWeight": "600"}),
                        ],
                        id="explore-load-container",
                        style={"display": "inline-flex", "alignItems": "center"},
                    ),
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
                    html.Div(
                        [
                            html.Button("Save changes", id="explore-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "16px"}),
                            html.Span(id="explore-save-status", style={"marginLeft": "8px", "fontWeight": "600"}),
                        ],
                        id="explore-save-container",
                        style={"display": "inline-flex", "alignItems": "center"},
                    ),
                ],
                style=CARD_STYLE,
                id="explore-table-container",
            ),
        ],
        style=PAGE_STYLE,
    )


def admin_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Admin", style={"margin": "0 0 8px 0", "color": "#0f172a", "fontSize": "28px"}),
                    html.P("Configure SQL warehouse, results table, volume path, and agent endpoint. Used by Ingest and Explore.", style={"margin": 0, "color": "#64748b"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3("Configuration", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    html.Label("SQL warehouse HTTP path", style=LABEL_STYLE),
                    dcc.Input(id="admin-http-path", type="text", placeholder=HTTP_PATH, style=INPUT_STYLE),
                    html.Label("Results table (catalog.schema.table)", style=LABEL_STYLE),
                    dcc.Input(id="admin-results-table", type="text", placeholder=RESULTS_TABLE, style=INPUT_STYLE),
                    html.Label("Volume path", style=LABEL_STYLE),
                    dcc.Input(id="admin-volume-path", type="text", placeholder=f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}", style=INPUT_STYLE),
                    html.Label("Agent endpoint name", style=LABEL_STYLE),
                    dcc.Input(id="admin-agent-endpoint", type="text", placeholder=AGENT_ENDPOINT, style=INPUT_STYLE),
                    html.Button("Save configuration", id="admin-save-btn", n_clicks=0, style={**BTN_PRIMARY, "marginTop": "8px"}),
                    html.Span(id="admin-save-status", style={"marginLeft": "12px", "color": "#059669", "fontWeight": "600"}),
                ],
                style=CARD_STYLE,
            ),
        ],
        style=PAGE_STYLE,
    )


@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/explore":
        return explore_layout()
    if pathname == "/admin":
        return admin_layout()
    return ingest_layout()


# ---------------------------------------------------------------------------
# Nav: three large tabs + logo (right), active tab from pathname
# ---------------------------------------------------------------------------
@callback(
    Output("nav-container", "children"),
    Input("url", "pathname"),
)
def render_nav(pathname):
    path = pathname or "/"
    tabs = [
        ("/", "Ingest"),
        ("/explore", "Explore"),
        ("/admin", "Admin"),
    ]
    tab_links = []
    for href, label in tabs:
        is_active = path == href
        tab_links.append(
            dcc.Link(label, href=href, style=_nav_tab_style(is_active), id=f"nav-{label.lower()}"),
        )
    logo_src = "/assets/vermeer-logo.png"
    # Grid: equal left/right columns so center (tabs) is truly centered; logo in right column so it never overlaps
    return html.Nav(
        [
            html.Div(style={"minWidth": "0"}),
            html.Div(
                tab_links,
                style={"display": "flex", "gap": "20px", "alignItems": "center", "justifyContent": "center", "minWidth": "0"},
            ),
            html.Div(
                html.Img(src=logo_src, alt="Vermeer", style={"height": "72px", "width": "auto", "display": "block", "marginLeft": "auto"}),
                style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "minWidth": "120px"},
            ),
        ],
        style={
            **NAV_STYLE,
            "display": "grid",
            "gridTemplateColumns": "1fr auto 1fr",
            "gap": "0",
            "alignItems": "center",
        },
    )


# ---------------------------------------------------------------------------
# Admin: sync store -> inputs when on Admin page; Save -> store
# ---------------------------------------------------------------------------
@callback(
    Output("admin-http-path", "value"),
    Output("admin-results-table", "value"),
    Output("admin-volume-path", "value"),
    Output("admin-agent-endpoint", "value"),
    Input("url", "pathname"),
    Input("app-config", "data"),
)
def admin_sync_inputs(pathname, config):
    if pathname != "/admin" or not config:
        return no_update, no_update, no_update, no_update
    return (
        config.get("http_path") or "",
        config.get("results_table") or "",
        config.get("volume_path") or "",
        config.get("agent_endpoint") or "",
    )


@callback(
    Output("app-config", "data"),
    Output("admin-save-status", "children"),
    Input("admin-save-btn", "n_clicks"),
    State("admin-http-path", "value"),
    State("admin-results-table", "value"),
    State("admin-volume-path", "value"),
    State("admin-agent-endpoint", "value"),
    State("app-config", "data"),
    prevent_initial_call=True,
)
def admin_save(n_clicks, http_path, results_table, volume_path, agent_endpoint, current):
    if not n_clicks:
        return no_update, ""
    current = current or _default_app_config()
    new_data = {
        "http_path": (http_path or "").strip() or current.get("http_path") or HTTP_PATH,
        "results_table": (results_table or "").strip() or current.get("results_table") or RESULTS_TABLE,
        "volume_path": (volume_path or "").strip() or current.get("volume_path") or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}",
        "agent_endpoint": (agent_endpoint or "").strip() or current.get("agent_endpoint") or AGENT_ENDPOINT,
    }
    return new_data, "✓ Saved!"


# ---------------------------------------------------------------------------
# Explore: fetch distinct filter options from results table when landing on Explore
# ---------------------------------------------------------------------------
@callback(
    Output("explore-filter-options", "data"),
    Input("url", "pathname"),
    State("app-config", "data"),
)
def fetch_explore_filter_options(pathname, config):
    if pathname != "/explore" or not config:
        return no_update
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    try:
        conn = get_connection(http_path)
        return read_distinct_filter_options(conn, table_name)
    except Exception as e:
        logger.warning("fetch_explore_filter_options: %s", e)
        return {"company": [], "number_of_cylinders": [], "vermeer_product": []}


@callback(
    Output("filter-manufacturer", "options"),
    Output("filter-cylinder-count", "options"),
    Output("filter-vermeer-product", "options"),
    Input("explore-filter-options", "data"),
    Input("url", "pathname"),
)
def explore_filter_dropdown_options(data, pathname):
    # Only update dropdowns when we're on Explore (avoids updating non-existent components)
    if pathname != "/explore":
        return no_update, no_update, no_update
    if not data:
        return no_update, no_update, no_update
    companies = data.get("company") or []
    cylinders = data.get("number_of_cylinders") or []
    products = data.get("vermeer_product") or []
    return (
        [{"label": str(v), "value": v} for v in companies],
        [{"label": str(v), "value": v} for v in cylinders],
        [{"label": str(v), "value": v} for v in products],
    )


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
        State("app-config", "data"),
    ],
    background=True,
    progress=[Output("ingest-progress", "children")],
    prevent_initial_call=True,
)
def run_ingest(set_progress, n_clicks, upload_data, config):
    empty_result = [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Select a PDF and click Parse & ingest.", step_tracker("")
    if not n_clicks or not upload_data or not upload_data.get("contents") or not upload_data.get("filename"):
        return empty_result
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    volume_path = (config.get("volume_path") or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/Volumes/"):
        parts = UPLOAD_VOLUME.split(".")
        volume_path = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    agent_endpoint = (config.get("agent_endpoint") or "").strip() or AGENT_ENDPOINT
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
        logger.info("Ingest: parse done, text length=%s", len(text))
        if not text.strip():
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Parsed document had no text content.", step_tracker("")

        set_progress(step_tracker("Extracting Information"))
        logger.info("Ingest: running ai_query for extraction (endpoint=%s)", agent_endpoint)
        agent_out = invoke_extraction_agent_sql(conn, filename, text, agent_endpoint)
        ingest_id = str(uuid.uuid4())
        today = date.today().isoformat()
        exploded = explode_agent_output(agent_out, today, filename, ingest_id)
        logger.info("Ingest: extraction done, exploded rows=%s", len(exploded))

        if len(exploded) == 0:
            logger.warning("Ingest: extraction returned 0 engine rows (agent_out keys=%s)", list(agent_out.keys()) if agent_out else "empty")
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Extraction returned 0 engine records. Check that the endpoint returns company/product_series/engines and that the document contains engine specs.", step_tracker("")

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
# Ingest: Save button (background so we can show "Saving" + pulse, then "✓ Saved!")
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("ingest-error", "children", allow_duplicate=True),
        Output("ingest-save-container", "children"),
    ),
    inputs=Input("ingest-save-btn", "n_clicks"),
    state=[
        State("ingest-table", "data"),
        State("ingest-table", "columns"),
        State("app-config", "data"),
    ],
    background=True,
    progress=[Output("ingest-save-container", "children")],
    prevent_initial_call=True,
)
def save_ingest(set_progress, n_clicks, data, columns, config):
    if not n_clicks or not data or not columns:
        return "No data to save.", _ingest_save_container("normal")
    set_progress(_ingest_save_container("saving"))
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    try:
        col_names = [c["id"] for c in columns]
        df = pd.DataFrame(data, columns=col_names)
        conn = get_connection(http_path)
        update_rows_by_id(conn, table_name, df)
        return "Saved successfully.", _ingest_save_container("saved")
    except Exception as e:
        logger.exception("save_ingest: %s", e)
        return f"Error saving: {e}", _ingest_save_container("normal")


# ---------------------------------------------------------------------------
# Explore: Load data (background so we can show "Loading" + pulse, then "✓ Loaded!")
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("explore-table", "data"),
        Output("explore-table", "columns"),
        Output("explore-error", "children"),
        Output("explore-load-container", "children"),
    ),
    inputs=[Input("explore-load-btn", "n_clicks"), Input("url", "pathname")],
    state=[
        State("app-config", "data"),
        State("filter-manufacturer", "value"),
        State("filter-ingest-date", "date"),
        State("filter-cylinder-count", "value"),
        State("filter-vermeer-product", "value"),
    ],
    background=True,
    progress=[Output("explore-load-container", "children")],
    prevent_initial_call=True,
)
def load_explore(set_progress, n_clicks, pathname, config, f_man, f_date, f_cyl, f_vermeer):
    empty = [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "", _explore_load_container("normal")
    # Load when user clicks Load or when navigating to Explore (so table shows data)
    if not n_clicks and pathname != "/explore":
        return empty
    set_progress(_explore_load_container("loading"))
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    parts = []
    if f_man and str(f_man).strip():
        parts.append(f"company = '{str(f_man).strip().replace(chr(39), chr(39)+chr(39))}'")
    if f_date:
        d_str = f_date if isinstance(f_date, str) else (f_date.strftime("%Y-%m-%d") if hasattr(f_date, "strftime") else str(f_date))
        if str(d_str).strip():
            parts.append(f"ingest_date = '{str(d_str).strip()}'")
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
        return data, cols, "", _explore_load_container("loaded")
    except Exception as e:
        logger.exception("load_explore: %s", e)
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e), _explore_load_container("normal")


# ---------------------------------------------------------------------------
# Explore: Save button (background so we can show "Saving" + pulse, then "✓ Saved!")
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("explore-error", "children", allow_duplicate=True),
        Output("explore-save-container", "children"),
    ),
    inputs=Input("explore-save-btn", "n_clicks"),
    state=[
        State("explore-table", "data"),
        State("explore-table", "columns"),
        State("app-config", "data"),
    ],
    background=True,
    progress=[Output("explore-save-container", "children")],
    prevent_initial_call=True,
)
def save_explore(set_progress, n_clicks, data, columns, config):
    if not n_clicks or not data or not columns:
        return "No data to save.", _explore_save_container("normal")
    set_progress(_explore_save_container("saving"))
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    try:
        col_names = [c["id"] for c in columns]
        df = pd.DataFrame(data, columns=col_names)
        conn = get_connection(http_path)
        update_rows_by_id(conn, table_name, df)
        return "Saved successfully.", _explore_save_container("saved")
    except Exception as e:
        logger.exception("save_explore: %s", e)
        return f"Error saving: {e}", _explore_save_container("normal")


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", "8080")))
    app.run(host="0.0.0.0", port=port, debug=False)
