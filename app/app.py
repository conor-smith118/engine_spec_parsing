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
import re
import time
import uuid

import requests
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
    ALL,
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
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "kie-c97b739c-endpoint")
# Knowledge Assistant endpoint for Explore tab chat (dashboard + chat view)
EXPLORE_KA_ENDPOINT = "ka-24b45243-endpoint"
# Published dashboard embedded on Explore tab (left side) - use /embed/ path for iframe
EXPLORE_DASHBOARD_URL = "https://adb-878896594214094.14.azuredatabricks.net/embed/dashboardsv3/01f110f755ac120886b8fced7e9b84f4?o=878896594214094"
# One-row table for ai_query input (same pattern as newly_parsed_temp -> ai_query in SQL)
EXTRACTION_INPUT_TABLE = "conor_smith.engine_specs_parse._extraction_input"
# Knowledge Assistant: sync after uploads to app_storage (token: KA_TOKEN env or dbutils.secrets css_tokens/ka_token)
APP_STORAGE_VOLUME_PATH = f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
KNOWLEDGE_ASSISTANT_ID = "24b45243-5459-43d0-9f91-43120316355e"

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


def _get_ka_token() -> str | None:
    """Knowledge Assistant API token: dbutils.secrets (Databricks runtime) or KA_TOKEN env."""
    try:
        from databricks.sdk.runtime import dbutils
        return dbutils.secrets.get(scope="css_tokens", key="ka_token")
    except Exception:
        return os.environ.get("KA_TOKEN")


def _sync_knowledge_assistant() -> str | None:
    """POST sync-knowledge-sources for the Knowledge Assistant. Returns None on success, error message string on failure."""
    token = _get_ka_token()
    if not token:
        return "Knowledge Assistant sync skipped: no KA_TOKEN or dbutils secret."
    host = (getattr(cfg, "host", None) or DATABRICKS_HOST or "").rstrip("/").replace("https://", "").replace("http://", "")
    if not host:
        return "Knowledge Assistant sync skipped: no workspace host."
    url = f"https://{host}/api/2.0/knowledge-assistants/{KNOWLEDGE_ASSISTANT_ID}/sync-knowledge-sources"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return None
    except requests.RequestException as e:
        msg = getattr(e, "response", None) and getattr(e.response, "text", None)
        return f"Knowledge Assistant sync failed: {e}" + (f" — {msg}" if msg else "")


def parse_agent_response(response: dict) -> tuple[str, list[dict]]:
    """Extract final answer and tool usage from Agent Bricks / Genie response. Returns (final_answer, tools_used)."""
    output = response.get("output", [])
    tools_used = []
    for item in output:
        if item.get("type") == "function_call":
            tools_used.append({
                "name": item.get("name"),
                "call_id": item.get("call_id"),
                "arguments": item.get("arguments"),
                "step": item.get("step"),
            })
    final_answer = None
    for item in reversed(output):
        if item.get("type") == "message" and item.get("status") == "completed":
            for content_item in (item.get("content") or []):
                if content_item.get("type") == "output_text":
                    final_answer = content_item.get("text", "")
                    break
            if final_answer:
                break
    if not final_answer:
        for item in reversed(output):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for content_item in (item.get("content") or []):
                    if content_item.get("type") == "output_text":
                        text = content_item.get("text", "")
                        if text and not (text.startswith("<name>") and text.endswith("</name>")):
                            final_answer = text
                            break
                if final_answer:
                    break
    return (final_answer or "No answer found", tools_used)


def invoke_explore_agent(messages: list[dict], temperature: float = 0.5) -> dict:
    """Call the Explore Knowledge Assistant endpoint. messages = [{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}, ...]."""
    w = get_workspace_client()
    return w.api_client.do(
        method="POST",
        path=f"/serving-endpoints/{EXPLORE_KA_ENDPOINT}/invocations",
        body={"input": messages, "temperature": temperature},
    )


# Grantee for volume_privileges (principal that has READ_VOLUME). Set VOLUME_GRANTEE env to override.
VOLUME_GRANTEE_DEFAULT = "0daada9f-9717-4b25-aefa-aee2033796b1"


def read_volumes_with_access(conn) -> list[dict]:
    """Return list of {volume_catalog, volume_schema, volume_name} for volumes the grantee can read."""
    grantee = os.environ.get("VOLUME_GRANTEE", VOLUME_GRANTEE_DEFAULT).strip()
    grantee_esc = grantee.replace("'", "''")
    q = f"""
    SELECT DISTINCT volume_catalog, volume_schema, volume_name
    FROM system.information_schema.volume_privileges
    WHERE grantee = '{grantee_esc}'
      AND privilege_type = 'READ_VOLUME'
    ORDER BY volume_catalog, volume_schema, volume_name
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(q)
            tbl = cursor.fetchall_arrow()
        if tbl is None or tbl.num_rows == 0:
            return []
        df = tbl.to_pandas()
        return df.to_dict("records")
    except Exception as e:
        logger.warning("read_volumes_with_access: %s", e)
        return []


def list_pdf_files_in_volume(volume_path: str) -> list[str]:
    """List all PDF filenames (relative to volume root) in a Unity Catalog volume, recursively."""
    w = get_workspace_client()
    base = volume_path.rstrip("/")
    out: list[str] = []

    def recurse(dir_path: str) -> None:
        path = f"{base}/{dir_path}" if dir_path else base
        try:
            # Use list_directory_contents (returns Iterator[DirectoryEntry]); DirectoryEntry has name, path, is_directory
            for entry in w.files.list_directory_contents(path):
                name = getattr(entry, "name", None) or ""
                p = getattr(entry, "path", None) or ""
                if not name:
                    continue
                is_dir = getattr(entry, "is_directory", False) or getattr(entry, "is_dir", False)
                if is_dir:
                    recurse(f"{dir_path}/{name}" if dir_path else name)
                elif name.lower().endswith(".pdf"):
                    # Store relative path for run_parse_document_sql(conn, volume_path, relative_path)
                    out.append(f"{dir_path}/{name}" if dir_path else name)
        except Exception as e:
            logger.warning("list_pdf_files_in_volume %s: %s", path, e)

    recurse("")
    return sorted(out)


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
    "id", "company", "product_series", "engine_part_number",
    "engine_type", "power_rating_continuous_operations", "number_of_cylinders",
    "specific_fuel_consumption", "ingest_date", "vermeer_product",
    "source_file", "ingest_id",
]


def ensure_results_table(conn, table_name: str) -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id STRING NOT NULL,
        company STRING,
        product_series STRING,
        engine_part_number STRING,
        engine_type STRING,
        power_rating_continuous_operations STRING,
        number_of_cylinders STRING,
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


def read_distinct_source_files(conn, table_name: str) -> list[str]:
    """Return distinct source_file values from the results table (for Explore chat document links)."""
    try:
        q = f"SELECT DISTINCT source_file FROM {table_name} WHERE source_file IS NOT NULL AND TRIM(source_file) != '' ORDER BY source_file"
        with conn.cursor() as cursor:
            cursor.execute(q)
            tbl = cursor.fetchall_arrow()
        if tbl is None or tbl.num_rows == 0:
            return []
        df = tbl.to_pandas()
        if df.empty or "source_file" not in df.columns:
            return []
        return df["source_file"].dropna().astype(str).str.strip().unique().tolist()
    except Exception as e:
        logger.warning("read_distinct_source_files: %s", e)
        return []


def source_file_exists_in_table(conn, table_name: str, source_file: str) -> bool:
    """Return True if the results table has at least one row with this source_file."""
    if not source_file or not str(source_file).strip():
        return False
    q = f"SELECT 1 FROM {table_name} WHERE source_file = {_format_sql_val(str(source_file).strip())} LIMIT 1"
    try:
        with conn.cursor() as cursor:
            cursor.execute(q)
            tbl = cursor.fetchall_arrow()
            return tbl is not None and tbl.num_rows > 0
    except Exception as e:
        logger.warning("source_file_exists_in_table: %s", e)
        return False


def delete_rows_by_source_file(conn, table_name: str, source_file: str) -> None:
    """Delete all rows in the results table where source_file matches."""
    if not source_file or not str(source_file).strip():
        return
    q = f"DELETE FROM {table_name} WHERE source_file = {_format_sql_val(str(source_file).strip())}"
    with conn.cursor() as cursor:
        cursor.execute(q)


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
    """Explode new extraction schema: { company, product_series, engines: [{ engine_part_number, number_of_cylinders, ... }] } -> one row per engine."""
    company = agent_output.get("company") or ""
    product_series = agent_output.get("product_series") or ""
    engines = agent_output.get("engines") or []
    rows = []
    for e in engines:
        cyl = e.get("number_of_cylinders")
        if cyl is not None and not isinstance(cyl, str):
            cyl = str(cyl)
        rows.append({
            "id": str(uuid.uuid4()),
            "company": company,
            "product_series": product_series,
            "engine_part_number": (e.get("engine_part_number") or ""),
            "engine_type": (e.get("engine_type") or ""),
            "power_rating_continuous_operations": (e.get("power_rating_continuous_operations") or ""),
            "number_of_cylinders": cyl or "",
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

# Shared styles (nav-container is just a wrapper; bar layout is built in render_nav)
NAV_STYLE = {
    "width": "100%",
    "minWidth": "0",
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
        base["backgroundColor"] = "#F7C632"
        base["color"] = "#1e293b"
        base["borderColor"] = "#F7C632"
    else:
        base["backgroundColor"] = "transparent"
        base["color"] = "#e2e8f0"
        base["borderColor"] = "#94a3b8"
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
                    html.H3("Ingest source", style={"margin": "0 0 16px 0", "fontSize": "16px", "color": "#334155"}),
                    dcc.RadioItems(
                        id="ingest-source-type",
                        options=[
                            {"label": "Upload file(s)", "value": "upload"},
                            {"label": "Ingest from volume", "value": "volume"},
                        ],
                        value="upload",
                        inline=True,
                        style={"marginBottom": "16px"},
                    ),
                    html.Div(
                        id="ingest-upload-section",
                        children=[
                            html.Label("Upload PDF(s)", style=LABEL_STYLE),
                            dcc.Upload(
                                id="ingest-upload",
                                children=html.Div(["Select PDF(s) or drag here"], style={"padding": "20px", "border": "2px dashed #cbd5e1", "borderRadius": "8px", "textAlign": "center", "color": "#64748b", "cursor": "pointer"}),
                                accept=".pdf",
                                multiple=True,
                            ),
                            html.Div(id="ingest-upload-filename", style={"marginTop": "8px", "fontSize": "14px", "color": "#475569"}),
                            html.Div(
                                [
                                    html.Label("Manufacturer", style={**LABEL_STYLE, "marginTop": "12px", "marginRight": "8px", "display": "inline-block"}),
                                    dcc.Input(id="ingest-manufacturer-override", type="text", placeholder="Optional: override company", style={**INPUT_STYLE, "width": "220px", "marginBottom": "0", "marginRight": "16px", "display": "inline-block"}),
                                    html.Label("Vermeer Product", style={**LABEL_STYLE, "marginRight": "8px", "display": "inline-block"}),
                                    dcc.Input(id="ingest-vermeer-product-override", type="text", placeholder="Optional: set Vermeer product", style={**INPUT_STYLE, "width": "220px", "marginBottom": "0", "display": "inline-block"}),
                                ],
                                style={"marginTop": "16px"},
                            ),
                            html.Div(
                                [
                                    html.Button("Parse & ingest", id="ingest-run-btn", n_clicks=0, style={**BTN_SUCCESS, "marginTop": "16px"}),
                                    html.Span(id="ingest-upload-to-volume-msg", style={"marginLeft": "12px", "fontSize": "14px", "color": "#64748b"}),
                                ],
                                style={"display": "inline-flex", "alignItems": "center"},
                            ),
                        ],
                    ),
                    html.Div(
                        id="ingest-volume-section",
                        style={"display": "none"},
                        children=[
                            html.Div(
                                [
                                    html.Label("Volume", style={**LABEL_STYLE, "marginTop": "0"}),
                                    dcc.Dropdown(id="ingest-volume-picker", options=[], placeholder="Select volume", style={"marginBottom": "8px"}),
                                ],
                            ),
                            html.Div(id="ingest-volume-files-info", style={"marginTop": "8px", "fontSize": "14px", "color": "#475569"}),
                            html.Div(
                                [
                                    html.Label("Manufacturer", style={**LABEL_STYLE, "marginTop": "12px", "marginRight": "8px", "display": "inline-block"}),
                                    dcc.Input(id="ingest-manufacturer-override-vol", type="text", placeholder="Optional: override company", style={**INPUT_STYLE, "width": "220px", "marginBottom": "0", "marginRight": "16px", "display": "inline-block"}),
                                    html.Label("Vermeer Product", style={**LABEL_STYLE, "marginRight": "8px", "display": "inline-block"}),
                                    dcc.Input(id="ingest-vermeer-product-override-vol", type="text", placeholder="Optional: set Vermeer product", style={**INPUT_STYLE, "width": "220px", "marginBottom": "0", "display": "inline-block"}),
                                ],
                                style={"marginTop": "12px"},
                            ),
                            html.Button("Parse & ingest volume", id="ingest-volume-run-btn", n_clicks=0, style={**BTN_SUCCESS, "marginTop": "16px"}, disabled=True),
                        ],
                    ),
                    html.Div(
                        id="ingest-already-ingested-container",
                        style={
                            "display": "none",
                            "marginTop": "16px",
                            "padding": "12px 16px",
                            "backgroundColor": "#fef3c7",
                            "borderRadius": "8px",
                            "border": "1px solid #f59e0b",
                        },
                        children=[],
                    ),
                    html.Div(id="ingest-progress", style={"marginTop": "20px"}),
                    html.Div(id="ingest-ka-sync-error", style={"color": "#dc2626", "marginTop": "8px", "fontSize": "14px"}),
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
            dcc.Store(id="ingest-upload-store", data={"files": []}),
            dcc.Store(id="ingest-upload-to-volume-status", data=None),
            dcc.Store(id="ingest-volume-files-store", data=None),
            dcc.Store(id="ingest-reingest-choices", data={"files": [], "actions": []}),
        ],
        style=PAGE_STYLE,
    )


def explore_layout():
    dashboard_iframe = html.Iframe(
        src=EXPLORE_DASHBOARD_URL,
        style={
            "width": "100%",
            "height": "600px",
            "border": "none",
            "borderRadius": "8px",
            "backgroundColor": "#fff",
        },
    )
    explore_tab_content = html.Div(
        [
            html.Div(
                [dashboard_iframe],
                style={"flex": "1", "minWidth": "0", "marginRight": "16px"},
            ),
            html.Div(
                [
                    html.H3("Knowledge Assistant", style={"margin": "0 0 8px 0", "fontSize": "16px", "color": "#334155"}),
                    html.P("Ask questions about engine specs.", style={"margin": "0 0 12px 0", "color": "#64748b", "fontSize": "14px"}),
                    html.Div(
                        id="explore-chat-messages",
                        style={
                            "minHeight": "200px",
                            "maxHeight": "400px",
                            "overflowY": "auto",
                            "padding": "12px",
                            "backgroundColor": "#f8fafc",
                            "borderRadius": "8px",
                            "border": "1px solid #e2e8f0",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="explore-chat-input",
                                type="text",
                                placeholder="e.g. Which engine has the highest rated power?",
                                style={**INPUT_STYLE, "marginBottom": "0", "marginRight": "8px", "flex": "1"},
                                debounce=False,
                            ),
                            html.Button("Send", id="explore-chat-send", n_clicks=0, style={**BTN_PRIMARY}),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                ],
                style={"width": "380px", "minWidth": "320px", "flexShrink": 0},
            ),
        ],
        style={"display": "flex", "gap": "16px", "alignItems": "stretch", "marginTop": "16px"},
    )
    edit_data_tab_content = html.Div(
        [
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
        style={"marginTop": "16px"},
    )
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Explore engine specs", style={"margin": "0 0 8px 0", "color": "#0f172a", "fontSize": "28px"}),
                    html.P("View the dashboard and chat with the Knowledge Assistant, or edit data in the Edit Data tab.", style={"margin": 0, "color": "#64748b"}),
                ],
                style=CARD_STYLE,
            ),
            dcc.Store(id="explore-chat-history", data=[]),
            dcc.Store(id="explore-chat-pending", data=None),
            dcc.Tabs(
                id="explore-page-tabs",
                value="explore",
                children=[
                    dcc.Tab(label="Explore", value="explore", children=explore_tab_content),
                    dcc.Tab(label="Edit Data", value="edit-data", children=edit_data_tab_content),
                ],
                style={"marginTop": "8px"},
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
    logo_src = "/assets/vermeer-logo2.png"
    # Single full-width grid: [1fr] [tabs] [1fr]. Equal side columns => tabs truly centered.
    # Logo lives in the right column so it never overlaps. Grid is the only child of nav-container
    # and must take full width (nav-container has width:100% so this fills it).
    return html.Div(
        [
            html.Div(style={"minWidth": "0"}),  # left column
            html.Div(
                tab_links,
                style={
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "minWidth": "0",
                },
            ),  # center column: tab links
            html.Div(
                html.Img(
                    src=logo_src,
                    alt="Vermeer",
                    style={"height": "72px", "width": "auto", "display": "block"},
                ),
                style={
                    "display": "flex",
                    "justifyContent": "flex-end",
                    "alignItems": "center",
                    "minWidth": "120px",
                },
            ),  # right column: logo right-aligned
        ],
        style={
            "width": "100%",
            "minWidth": "0",
            "boxSizing": "border-box",
            "display": "grid",
            "gridTemplateColumns": "1fr auto 1fr",
            "gap": "0",
            "alignItems": "center",
            "padding": "12px 24px",
            "backgroundColor": "#005C29",
            "borderBottom": "1px solid #e2e8f0",
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
def _assistant_content_with_links(content: str) -> list:
    """Parse assistant content: turn markdown [label](url) and raw PDF URLs into clickable links that open in a new tab."""
    if not (content or isinstance(content, str)):
        return [""]
    content = content.strip() or "(empty)"
    parts = []
    last_end = 0
    # Markdown links [label](url) where url contains .pdf
    md_pattern = re.compile(r"\[([^\]]*)\]\((https://[^)]+)\)", re.IGNORECASE)
    for m in md_pattern.finditer(content):
        label, url = m.group(1).strip(), m.group(2)
        if ".pdf" in url.lower():
            if last_end < m.start():
                parts.append(content[last_end : m.start()])
            link_label = label or "PDF"
            parts.append(
                html.A(
                    link_label,
                    href=url,
                    target="_blank",
                    rel="noopener noreferrer",
                    style={"color": "inherit", "textDecoration": "underline"},
                )
            )
            last_end = m.end()
    parts.append(content[last_end:])
    # If no links were found, return single string for the bubble
    if len(parts) == 1 and all(isinstance(p, str) for p in parts):
        return [parts[0]] if parts[0] else [""]
    # Keep non-empty text segments and all link components
    return [p for p in parts if (isinstance(p, str) and p) or not isinstance(p, str)]


def _chat_message_bubbles(history: list) -> list:
    """Build list of message bubble Divs from chat history [{role, content}, ...]. Assistant content may contain PDF links that open in new tab."""
    out = []
    for msg in history or []:
        role = (msg.get("role") or "user").lower()
        content = (msg.get("content") or "").strip() or "(empty)"
        is_user = role == "user"
        bubble_style = {
            "padding": "10px 14px",
            "borderRadius": "12px",
            "maxWidth": "85%",
            "whiteSpace": "pre-wrap",
            "wordBreak": "break-word",
            "fontSize": "14px",
            "backgroundColor": "#e2e8f0" if is_user else "#2563eb",
            "color": "#0f172a" if is_user else "#ffffff",
        }
        if is_user:
            inner_children = content
        else:
            inner_children = _assistant_content_with_links(content)
        out.append(
            html.Div(
                html.Div(inner_children, style=bubble_style),
                style={
                    "display": "flex",
                    "justifyContent": "flex-end" if is_user else "flex-start",
                    "marginBottom": "8px",
                },
            )
        )
    return out


def _chat_loading_throbber():
    """Loading indicator shown under the last message while the agent is responding."""
    return html.Div(
        html.Div(className="chat-loading-throbber"),
        style={
            "display": "flex",
            "justifyContent": "flex-start",
            "marginBottom": "8px",
            "marginLeft": "4px",
        },
    )


@callback(
    Output("explore-chat-history", "data"),
    Output("explore-chat-messages", "children"),
    Output("explore-chat-input", "value"),
    Output("explore-chat-pending", "data"),
    Input("explore-chat-send", "n_clicks"),
    State("explore-chat-input", "value"),
    State("explore-chat-history", "data"),
    prevent_initial_call=True,
)
def explore_chat_send_immediate(n_clicks, user_text, history):
    """Show user message and loading throbber immediately; trigger background callback to get answer."""
    if not n_clicks or not (user_text and str(user_text).strip()):
        return no_update, no_update, no_update, no_update
    history = list(history or [])
    user_content = str(user_text).strip()
    history.append({"role": "user", "content": user_content})
    bubbles = _chat_message_bubbles(history)
    pending = {"history": history}
    return history, bubbles + [_chat_loading_throbber()], "", pending


def _strip_footnotes_and_citations(text: str) -> str:
    """Remove inline citation refs [^ref] and entire footnote definition blocks [^ref]: ... from agent output."""
    if not text or not isinstance(text, str):
        return text
    # Remove inline citation refs e.g. [^XoOB-1], [^XoOB-2]
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    # Keep only content before the first footnote definition block (e.g. "[^XoOB-1]: General Engine Data...")
    match = re.search(r"\n\s*\[\^[^\]]+\]:", text)
    if match:
        text = text[: match.start()]
    return text.strip()


def _append_related_document_links(
    assistant_content: str,
    final_answer: str,
    config: dict | None,
) -> str:
    """If the response has no PDF link but mentions document(s) from the results table, append Related document(s) links at the end."""
    if not assistant_content or not config:
        return assistant_content
    if re.search(r"\]\(https?://[^)]+\.pdf", assistant_content, re.IGNORECASE):
        return assistant_content
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    volume_path = (config.get("volume_path") or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/"):
        volume_path = "/" + volume_path
    host = (getattr(cfg, "host", None) or DATABRICKS_HOST or "").rstrip("/")
    if not host:
        return assistant_content
    base_url = host + "/ajax-api/2.0/fs/files" + volume_path.rstrip("/")
    try:
        conn = get_connection(http_path)
        source_files = read_distinct_source_files(conn, table_name)
    except Exception as e:
        logger.warning("_append_related_document_links: %s", e)
        return assistant_content
    text_lower = (final_answer or "").lower()
    mentioned = []
    for fn in source_files:
        if not fn:
            continue
        name_without_ext = fn[:-4] if fn.lower().endswith(".pdf") else fn
        if name_without_ext.lower() in text_lower or fn.lower() in text_lower:
            mentioned.append(fn)
    if not mentioned:
        return assistant_content
    links = "\n".join(f"[{fn}]({base_url}/{fn})" for fn in mentioned)
    return assistant_content + "\n\nRelated document(s):\n" + links


@callback(
    Output("explore-chat-history", "data", allow_duplicate=True),
    Output("explore-chat-messages", "children", allow_duplicate=True),
    Output("explore-chat-pending", "data", allow_duplicate=True),
    Input("explore-chat-pending", "data"),
    State("app-config", "data"),
    background=True,
    prevent_initial_call=True,
)
def explore_chat_agent_response(pending, config):
    """Background: call agent, then replace throbber with formatted answer (Tools Used + answer). Append Related document(s) links when response references docs but has no PDF link."""
    if not pending or not isinstance(pending, dict) or not pending.get("history"):
        return no_update, no_update, no_update
    history = list(pending["history"])
    try:
        response = invoke_explore_agent(history, temperature=0.5)
        final_answer, tools_used = parse_agent_response(response)
    except Exception as e:
        logger.exception("explore_chat_agent_response: %s", e)
        final_answer = f"Sorry, an error occurred: {e}"
        tools_used = []
    tool_names = [t.get("name") or "Unknown" for t in tools_used if t.get("name")]
    tools_line = "Tools Used: " + (", ".join(tool_names) if tool_names else "None")
    answer_clean = _strip_footnotes_and_citations(final_answer or "No answer found")
    assistant_content = tools_line + "\n\n" + answer_clean
    assistant_content = _append_related_document_links(assistant_content, answer_clean, config)
    history.append({"role": "assistant", "content": assistant_content})
    bubbles = _chat_message_bubbles(history)
    return history, bubbles, None


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
# Ingest: store upload (supports single or multiple files)
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
    # dcc.Upload with multiple=True returns list of contents and list of filenames
    if isinstance(contents, list):
        filenames = filename if isinstance(filename, list) else [filename] * len(contents)
        files = [{"contents": c, "filename": (f or f"file_{i}.pdf")} for i, (c, f) in enumerate(zip(contents, filenames))]
    else:
        files = [{"contents": contents, "filename": (filename or "document.pdf")}]
    if not files:
        return no_update, ""
    labels = [f["filename"] for f in files]
    return {"files": files}, html.Span(f"Selected: {', '.join(labels)}" if len(labels) <= 3 else f"Selected: {len(labels)} file(s)")


# ---------------------------------------------------------------------------
# Ingest: set "uploading" as soon as files are selected (so button stays disabled until upload-to-volume completes)
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-upload-to-volume-status", "data"),
    Input("ingest-upload-store", "data"),
)
def ingest_upload_status_uploading(upload_data):
    files = (upload_data or {}).get("files") or []
    if not files:
        return None
    return "uploading"


# ---------------------------------------------------------------------------
# Ingest: upload selected files to volume in background; when done set status "ready" and sync Knowledge Assistant if app_storage
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("ingest-upload-to-volume-status", "data", allow_duplicate=True),
        Output("ingest-ka-sync-error", "children"),
    ),
    inputs=Input("ingest-upload-store", "data"),
    state=[State("app-config", "data")],
    background=True,
    prevent_initial_call=True,
)
def ingest_upload_to_volume(upload_data, config):
    files = (upload_data or {}).get("files") or []
    if not files:
        return None, ""
    config = config or _default_app_config()
    volume_path = (config.get("volume_path") or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/Volumes/"):
        parts = UPLOAD_VOLUME.split(".")
        volume_path = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    w = get_workspace_client()
    filenames = []
    for file_item in files:
        filename = (file_item.get("filename") or "document.pdf").strip()
        content_str = file_item.get("contents") or ""
        if "," in content_str:
            content_str = content_str.split(",", 1)[1]
        try:
            file_bytes = base64.b64decode(content_str)
        except Exception:
            continue
        upload_path = f"{volume_path.rstrip('/')}/{filename}"
        try:
            w.files.upload(upload_path, io.BytesIO(file_bytes), overwrite=True)
            filenames.append(filename)
        except Exception as e:
            logger.warning("ingest_upload_to_volume: %s for %s", e, filename)
    status_data = {"status": "ready", "filenames": filenames} if filenames else None
    ka_error = ""
    if filenames and volume_path.rstrip("/") == APP_STORAGE_VOLUME_PATH.rstrip("/"):
        err = _sync_knowledge_assistant()
        if err:
            ka_error = err
            logger.warning("ingest_upload_to_volume: %s", err)
    return status_data, ka_error


# ---------------------------------------------------------------------------
# Ingest: disable Parse & ingest until files are uploaded to volume
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-run-btn", "disabled"),
    Output("ingest-upload-to-volume-msg", "children"),
    Input("ingest-upload-store", "data"),
    Input("ingest-upload-to-volume-status", "data"),
)
def ingest_run_btn_disabled(upload_data, upload_status):
    files = (upload_data or {}).get("files") or []
    if not files:
        return True, ""
    if upload_status is None or upload_status == "uploading":
        return True, "Uploading to volume…"
    if isinstance(upload_status, dict) and upload_status.get("status") == "ready":
        return False, ""
    return True, ""


# ---------------------------------------------------------------------------
# Ingest: show "File already ingested" warning + re-ingest options when source_file exists in results table
# ---------------------------------------------------------------------------
BASE_ALREADY_INGESTED_STYLE = {
    "marginTop": "16px",
    "padding": "12px 16px",
    "backgroundColor": "#fef3c7",
    "borderRadius": "8px",
    "border": "1px solid #f59e0b",
}


@callback(
    Output("ingest-already-ingested-container", "style"),
    Output("ingest-already-ingested-container", "children"),
    Output("ingest-reingest-choices", "data"),
    Input("ingest-upload-store", "data"),
    Input("ingest-source-type", "value"),
    Input("ingest-volume-files-store", "data"),
    State("app-config", "data"),
)
def ingest_show_already_ingested_warning(upload_data, source_type, volume_files, config):
    empty_style = {**BASE_ALREADY_INGESTED_STYLE, "display": "none"}
    empty_store = {"files": [], "actions": []}
    # Build list of filenames to check (from upload or volume)
    if source_type == "volume":
        filenames = [f.strip() for f in (volume_files or []) if f and str(f).strip()]
    else:
        files = (upload_data or {}).get("files") or []
        filenames = [(f.get("filename") or "").strip() for f in files if (f.get("filename") or "").strip()]
    if not filenames:
        return empty_style, [], empty_store
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    try:
        conn = get_connection(http_path)
        already = [fn for fn in filenames if source_file_exists_in_table(conn, table_name, fn)]
    except Exception as e:
        logger.warning("ingest_show_already_ingested_warning: %s", e)
        return empty_style, [], empty_store
    if not already:
        return empty_style, [], empty_store
    # One row per file with Overwrite / Duplicate / Ignore
    action_options = [
        {"label": "Overwrite", "value": "overwrite"},
        {"label": "Duplicate", "value": "duplicate"},
        {"label": "Ignore", "value": "ignore"},
    ]
    children = [
        html.P("The following files are already ingested. Choose an action for each:", style={"margin": "0 0 12px 0", "color": "#b45309", "fontWeight": "600", "fontSize": "14px"}),
    ]
    for i, fn in enumerate(already):
        children.append(
            html.Div(
                [
                    html.Span(fn, style={"marginRight": "12px", "fontSize": "13px", "minWidth": "180px", "display": "inline-block"}),
                    dcc.Dropdown(
                        id={"type": "ingest-file-action", "index": i},
                        options=action_options,
                        value="overwrite",
                        clearable=False,
                        style={"width": "140px", "display": "inline-block", "fontSize": "13px"},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
        )
    return {**BASE_ALREADY_INGESTED_STYLE, "display": "block"}, children, {"files": already, "actions": ["overwrite"] * len(already)}


@callback(
    Output("ingest-reingest-choices", "data", allow_duplicate=True),
    Input({"type": "ingest-file-action", "index": ALL}, "value"),
    State("ingest-reingest-choices", "data"),
    prevent_initial_call=True,
)
def ingest_update_reingest_choices(values, current):
    if not current or not current.get("files"):
        return no_update
    # values are in index order (one per dropdown)
    n = len(current["files"])
    actions = []
    for i, v in enumerate(values or []):
        if i >= n:
            break
        actions.append(v if v in ("overwrite", "duplicate", "ignore") else "overwrite")
    while len(actions) < n:
        actions.append("overwrite")
    return {"files": current["files"], "actions": actions}


# ---------------------------------------------------------------------------
# Ingest: show only the selected source section (upload vs volume)
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-upload-section", "style"),
    Output("ingest-volume-section", "style"),
    Input("ingest-source-type", "value"),
)
def ingest_toggle_source_section(source_type):
    if source_type == "volume":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# ---------------------------------------------------------------------------
# Ingest: Load volumes when user selects "Ingest from volume" (no button)
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-volume-picker", "options"),
    Input("ingest-source-type", "value"),
    State("app-config", "data"),
)
def ingest_load_volumes(source_type, config):
    if source_type != "volume" or not config:
        return no_update
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    try:
        conn = get_connection(http_path)
        vols = read_volumes_with_access(conn)
        return [{"label": f"{v['volume_catalog']}.{v['volume_schema']}.{v['volume_name']}", "value": f"/Volumes/{v['volume_catalog']}/{v['volume_schema']}/{v['volume_name']}"} for v in vols]
    except Exception as e:
        logger.exception("ingest_load_volumes: %s", e)
        return []


# ---------------------------------------------------------------------------
# Ingest: When volume selected, list PDFs and enable Parse & ingest volume button
# ---------------------------------------------------------------------------
@callback(
    Output("ingest-volume-files-store", "data"),
    Output("ingest-volume-files-info", "children"),
    Output("ingest-volume-run-btn", "disabled"),
    Input("ingest-volume-picker", "value"),
)
def ingest_volume_selected(volume_path):
    if not volume_path or not str(volume_path).strip():
        return None, "", True
    try:
        pdfs = list_pdf_files_in_volume(volume_path)
        if not pdfs:
            return [], html.Span("No PDF files in this volume.", style={"color": "#64748b"}), True
        return pdfs, html.Span(f"{len(pdfs)} PDF(s) found. Click Parse & ingest volume to process.", style={"color": "#059669"}), False
    except Exception as e:
        logger.exception("ingest_volume_selected: %s", e)
        return None, html.Span(f"Error listing volume: {e}", style={"color": "#dc2626"}), True


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
        State("ingest-upload-to-volume-status", "data"),
        State("app-config", "data"),
        State("ingest-reingest-choices", "data"),
        State("ingest-manufacturer-override", "value"),
        State("ingest-vermeer-product-override", "value"),
    ],
    background=True,
    progress=[Output("ingest-progress", "children")],
    prevent_initial_call=True,
)
def run_ingest(set_progress, n_clicks, upload_data, upload_status, config, reingest_choices, manufacturer_override, vermeer_product_override):
    empty_result = [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Select PDF(s) and click Parse & ingest.", step_tracker("")
    files = (upload_data or {}).get("files") or []
    if not n_clicks or not files:
        return empty_result
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    volume_path = (config.get("volume_path") or "").strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
    if not volume_path.startswith("/Volumes/"):
        parts = UPLOAD_VOLUME.split(".")
        volume_path = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    agent_endpoint = (config.get("agent_endpoint") or "").strip() or AGENT_ENDPOINT
    conn = get_connection(http_path)
    ensure_results_table(conn, table_name)
    total = len(files)
    ingested_ids: list[str] = []
    file_status: list[tuple[str, str]] = []  # (filename, "completed" | "ignored" | "skipped")

    def _progress_ui(step_el, current_file: str):
        log = [html.P(f"File {idx + 1} of {total}: {current_file}", style={"margin": "8px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"})]
        for fn, status in file_status:
            color = "#059669" if status == "completed" else "#64748b" if status == "ignored" else "#f59e0b"
            log.append(html.Div(f"  {fn}: {status}", style={"fontSize": "12px", "color": color, "marginLeft": "8px"}))
        return html.Div([step_el] + log)

    try:
        for idx, file_item in enumerate(files):
            filename = (file_item.get("filename") or "document.pdf").strip()
            action = "duplicate"
            if reingest_choices and reingest_choices.get("files") and filename in reingest_choices["files"]:
                idx_choice = reingest_choices["files"].index(filename)
                action = (reingest_choices.get("actions") or [])[idx_choice] if idx_choice < len(reingest_choices.get("actions") or []) else "duplicate"
            if action == "ignore":
                file_status.append((filename, "ignored"))
                set_progress(_progress_ui(step_tracker("Done"), filename))
                continue

            set_progress(_progress_ui(step_tracker("Parsing Document"), filename))
            already_on_volume = (
                isinstance(upload_status, dict)
                and upload_status.get("status") == "ready"
                and filename in (upload_status.get("filenames") or [])
            )
            if not already_on_volume:
                content_str = file_item.get("contents") or ""
                if "," in content_str:
                    content_str = content_str.split(",", 1)[1]
                file_bytes = base64.b64decode(content_str)
                w = get_workspace_client()
                upload_path = f"{volume_path.rstrip('/')}/{filename}"
                w.files.upload(upload_path, io.BytesIO(file_bytes), overwrite=True)

            set_progress(_progress_ui(step_tracker("Parsing Document"), filename))
            parsed = run_parse_document_sql(conn, volume_path, filename)
            if not parsed:
                file_status.append((filename, "skipped (no parse)"))
                set_progress(_progress_ui(step_tracker("Done"), filename))
                return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], f"No parsed document for {filename}.", step_tracker("Done")
            text = extract_text_from_parsed(parsed)
            if not text.strip():
                file_status.append((filename, "skipped (no text)"))
                set_progress(_progress_ui(step_tracker("Done"), filename))
                return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], f"Parsed document had no text: {filename}.", step_tracker("Done")

            set_progress(_progress_ui(step_tracker("Extracting Information"), filename))
            agent_out = invoke_extraction_agent_sql(conn, filename, text, agent_endpoint)
            ingest_id = str(uuid.uuid4())
            ingested_ids.append(ingest_id)
            today = date.today().isoformat()
            exploded = explode_agent_output(agent_out, today, filename, ingest_id)
            if manufacturer_override and str(manufacturer_override).strip():
                exploded["company"] = str(manufacturer_override).strip()
            if vermeer_product_override and str(vermeer_product_override).strip():
                exploded["vermeer_product"] = str(vermeer_product_override).strip()
            if len(exploded) == 0:
                logger.warning("Ingest: no engine rows for %s", filename)
                file_status.append((filename, "skipped (no engine data)"))
                set_progress(_progress_ui(step_tracker("Done"), filename))
                continue
            set_progress(_progress_ui(step_tracker("Writing Data"), filename))
            if action == "overwrite" and source_file_exists_in_table(conn, table_name, filename):
                delete_rows_by_source_file(conn, table_name, filename)
            insert_rows(conn, table_name, exploded)
            file_status.append((filename, "completed"))

        set_progress(html.Div([step_tracker("Done")] + [html.Div(f"  {fn}: {st}", style={"fontSize": "12px", "color": "#059669" if st == "completed" else "#64748b" if st == "ignored" else "#f59e0b", "marginLeft": "8px"}) for fn, st in file_status]))
        if not ingested_ids:
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "No engine records extracted from any file.", step_tracker("Done")
        where = " OR ".join(f"ingest_id = '{i.replace(chr(39), chr(39)+chr(39))}'" for i in ingested_ids)
        df = read_table(conn, table_name, where)
        data = df.astype(str).replace("nan", "").to_dict("records")
        cols = [{"name": c, "id": c} for c in RESULTS_COLUMNS]
        return data, cols, "", step_tracker("Done")
    except Exception as e:
        logger.exception("run_ingest: %s", e)
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e), step_tracker("")


# ---------------------------------------------------------------------------
# Ingest: Parse & ingest volume (background; process each PDF in the selected volume)
# ---------------------------------------------------------------------------
@callback(
    output=(
        Output("ingest-table", "data", allow_duplicate=True),
        Output("ingest-table", "columns", allow_duplicate=True),
        Output("ingest-error", "children", allow_duplicate=True),
        Output("ingest-progress", "children", allow_duplicate=True),
    ),
    inputs=Input("ingest-volume-run-btn", "n_clicks"),
    state=[
        State("ingest-volume-picker", "value"),
        State("ingest-volume-files-store", "data"),
        State("app-config", "data"),
        State("ingest-reingest-choices", "data"),
        State("ingest-manufacturer-override-vol", "value"),
        State("ingest-vermeer-product-override-vol", "value"),
    ],
    background=True,
    progress=[Output("ingest-progress", "children", allow_duplicate=True)],
    prevent_initial_call=True,
)
def run_volume_ingest(set_progress, n_clicks, volume_path, file_list, config, reingest_choices, manufacturer_override, vermeer_product_override):
    empty_result = [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "Select a volume and click Parse & ingest volume.", step_tracker("")
    if not n_clicks or not volume_path or not file_list:
        return empty_result
    config = config or _default_app_config()
    http_path = (config.get("http_path") or "").strip() or HTTP_PATH
    table_name = (config.get("results_table") or "").strip() or RESULTS_TABLE
    agent_endpoint = (config.get("agent_endpoint") or "").strip() or AGENT_ENDPOINT
    conn = get_connection(http_path)
    ensure_results_table(conn, table_name)
    total = len(file_list)
    ingested_ids: list[str] = []
    file_status: list[tuple[str, str]] = []  # (filename, "completed" | "ignored" | "skipped")

    def _vol_progress_ui(step_el, current_file: str):
        log = [html.P(f"File {idx + 1} of {total}: {current_file}", style={"margin": "8px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"})]
        for fn, status in file_status:
            color = "#059669" if status == "completed" else "#64748b" if status == "ignored" else "#f59e0b"
            log.append(html.Div(f"  {fn}: {status}", style={"fontSize": "12px", "color": color, "marginLeft": "8px"}))
        return html.Div([step_el] + log)

    try:
        for idx, filename in enumerate(file_list):
            filename = (filename or "").strip()
            if not filename.lower().endswith(".pdf"):
                continue
            action = "duplicate"
            if reingest_choices and reingest_choices.get("files") and filename in reingest_choices["files"]:
                idx_choice = reingest_choices["files"].index(filename)
                action = (reingest_choices.get("actions") or [])[idx_choice] if idx_choice < len(reingest_choices.get("actions") or []) else "duplicate"
            if action == "ignore":
                file_status.append((filename, "ignored"))
                set_progress(_vol_progress_ui(step_tracker("Done"), filename))
                continue

            set_progress(_vol_progress_ui(step_tracker("Parsing Document"), filename))
            parsed = run_parse_document_sql(conn, volume_path, filename)
            if not parsed:
                logger.warning("run_volume_ingest: no parsed document for %s", filename)
                file_status.append((filename, "skipped (no parse)"))
                set_progress(_vol_progress_ui(step_tracker("Done"), filename))
                continue
            text = extract_text_from_parsed(parsed)
            if not text.strip():
                file_status.append((filename, "skipped (no text)"))
                set_progress(_vol_progress_ui(step_tracker("Done"), filename))
                continue
            set_progress(_vol_progress_ui(step_tracker("Extracting Information"), filename))
            agent_out = invoke_extraction_agent_sql(conn, filename, text, agent_endpoint)
            ingest_id = str(uuid.uuid4())
            ingested_ids.append(ingest_id)
            today = date.today().isoformat()
            exploded = explode_agent_output(agent_out, today, filename, ingest_id)
            if manufacturer_override and str(manufacturer_override).strip():
                exploded["company"] = str(manufacturer_override).strip()
            if vermeer_product_override and str(vermeer_product_override).strip():
                exploded["vermeer_product"] = str(vermeer_product_override).strip()
            if len(exploded) == 0:
                file_status.append((filename, "skipped (no engine data)"))
                set_progress(_vol_progress_ui(step_tracker("Done"), filename))
                continue
            set_progress(_vol_progress_ui(step_tracker("Writing Data"), filename))
            if action == "overwrite" and source_file_exists_in_table(conn, table_name, filename):
                delete_rows_by_source_file(conn, table_name, filename)
            insert_rows(conn, table_name, exploded)
            file_status.append((filename, "completed"))

        set_progress(html.Div([step_tracker("Done")] + [html.Div(f"  {fn}: {st}", style={"fontSize": "12px", "color": "#059669" if st == "completed" else "#64748b" if st == "ignored" else "#f59e0b", "marginLeft": "8px"}) for fn, st in file_status]))
        if not ingested_ids:
            return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], "No engine records extracted from any file in the volume.", html.Div([step_tracker("Done")] + [html.Div(f"  {fn}: {st}", style={"fontSize": "12px", "color": "#059669" if st == "completed" else "#64748b" if st == "ignored" else "#f59e0b", "marginLeft": "8px"}) for fn, st in file_status])
        where = " OR ".join(f"ingest_id = '{i.replace(chr(39), chr(39)+chr(39))}'" for i in ingested_ids)
        df = read_table(conn, table_name, where)
        data = df.astype(str).replace("nan", "").to_dict("records")
        cols = [{"name": c, "id": c} for c in RESULTS_COLUMNS]
        return data, cols, "", step_tracker("Done")
    except Exception as e:
        logger.exception("run_volume_ingest: %s", e)
        done_ui = html.Div([step_tracker("Done")] + [html.Div(f"  {fn}: {st}", style={"fontSize": "12px", "color": "#059669" if st == "completed" else "#64748b" if st == "ignored" else "#f59e0b", "marginLeft": "8px"}) for fn, st in file_status]) if file_status else step_tracker("Done")
        return [], [{"name": c, "id": c} for c in RESULTS_COLUMNS], str(e), done_ui


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
