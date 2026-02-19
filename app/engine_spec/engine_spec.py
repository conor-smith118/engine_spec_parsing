"""
Engine Spec Parsing – Databricks Reflex app.

Page 1 (Ingest): Upload PDF → ai_parse_document → Information Extraction Agent
  → explode JSON → append to results table → editable table → Save.
Page 2 (Explore): Filters (Manufacturer, Ingest Date, Cylinder count, Vermeer Product)
  → load into editable table → view/update/save rows.
"""
import io
import json
import logging
import os
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd
import reflex as rx
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (override via env in Databricks)
# ---------------------------------------------------------------------------
HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH", "")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "main.default.engine_spec_results")
UPLOAD_VOLUME = os.environ.get("UPLOAD_VOLUME", "main.default.uploads")
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "kie-c2e65325-endpoint")

# ---------------------------------------------------------------------------
# SQL connection (cookbook pattern)
# ---------------------------------------------------------------------------
_connection = None
_w: WorkspaceClient | None = None


def get_workspace_client() -> WorkspaceClient:
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w


def get_connection(http_path: str):
    global _connection
    if _connection is not None:
        return _connection
    cfg = Config()
    _connection = sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=cfg.authenticate,
    )
    return _connection


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
    "id",
    "company",
    "product_series",
    "engine_type",
    "power_rating_continuous_operations",
    "number_of_cylinders",
    "specific_fuel_consumption",
    "ingest_date",
    "vermeer_product",
    "source_file",
    "ingest_id",
]


def ensure_results_table(conn, table_name: str) -> None:
    """Create the results table if it does not exist."""
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


def pandas_to_editor_format(
    df: pd.DataFrame,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """Convert DataFrame to (rows, columns) for rx.data_editor."""
    if df.empty:
        return [], []
    # Ensure all columns are strings for display/edit
    df = df.astype(str).replace("nan", "")
    data = df.values.tolist()
    columns = [
        {"title": col, "id": col, "type": "str"}
        for col in df.columns
    ]
    return data, columns


def update_rows_by_id(
    conn, table_name: str, df: pd.DataFrame, id_col: str = "id"
) -> None:
    """Update existing rows by id. df must contain id and all columns to set."""
    if df.empty:
        return
    col_names = [c for c in df.columns if c != id_col]
    for _, row in df.iterrows():
        row_id = row[id_col]
        set_parts = ", ".join(
            f"{c} = {_format_sql_val(row[c])}" for c in col_names
        )
        sql_update = f"UPDATE {table_name} SET {set_parts} WHERE id = {_format_sql_val(str(row_id))}"
        with conn.cursor() as cursor:
            cursor.execute(sql_update)


def insert_rows(conn, table_name: str, df: pd.DataFrame) -> None:
    """Insert rows. df must have columns matching RESULTS_COLUMNS."""
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
    """Build full text from ai_parse_document VARIANT output."""
    try:
        doc = parsed.get("document") or {}
        elements = doc.get("elements") or []
        parts = []
        for el in elements:
            content = el.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        return "\n\n".join(parts)
    except Exception as e:
        logger.exception("extract_text_from_parsed: %s", e)
        return ""


def invoke_extraction_agent(text: str, endpoint: str) -> dict:
    """Call the Information Extraction Agent and return parsed JSON."""
    w = get_workspace_client()
    # Try common payload shapes for agent/LLM endpoints
    last_error = None
    for payload in (
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
    # Try to parse JSON from response (may be wrapped)
    try:
        if isinstance(raw, str) and raw.strip().startswith("{"):
            return json.loads(raw)
        # Some endpoints return list with one string
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, dict) else {}
    except json.JSONDecodeError:
        # Try to find JSON block in text
        start = raw.find("{")
        if start >= 0:
            end = raw.rfind("}") + 1
            if end > start:
                return json.loads(raw[start:end])
        raise


def explode_agent_output(
    agent_output: dict,
    ingest_date: str,
    source_file: str,
    ingest_id: str,
) -> pd.DataFrame:
    """Turn agent JSON into one row per engine with company, product_series, engine fields."""
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
    """Run READ_FILES + ai_parse_document and return parsed VARIANT as dict."""
    # Path like /Volumes/catalog/schema/volume/filename.pdf
    full_path = f"{volume_path.rstrip('/')}/{file_name}"
    # Escape single quotes in path
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
# State: Ingest page
# ---------------------------------------------------------------------------
class IngestState(rx.State):
    http_path: str = ""
    table_name: str = RESULTS_TABLE
    volume_path: str = ""  # e.g. /Volumes/main/default/uploads
    agent_endpoint: str = AGENT_ENDPOINT

    uploaded_filename: str = ""
    ingest_id: str = ""
    df_data: list[list[Any]] = []
    df_columns: list[dict] = []
    is_loading: bool = False
    is_saving: bool = False
    error_message: str = ""

    @rx.var
    def columns_for_editor(self) -> list[dict]:
        return self.df_columns

    @rx.var
    def data_for_editor(self) -> list[list[Any]]:
        return self.df_data

    @rx.event
    def handle_cell_change(self, cell: tuple[int, int], new_value: Any):
        # on_cell_edited passes (cell, new_value); cell is (col_index, row_index)
        col_index, row_index = cell[0], cell[1]
        val = getattr(new_value, "data", new_value)
        if 0 <= row_index < len(self.df_data) and 0 <= col_index < len(self.df_data[row_index]):
            self.df_data[row_index][col_index] = val

    @rx.event(background=True)
    async def run_ingest(self, files: list[rx.UploadFile]):
        if not files:
            return
        file = files[0]
        async with self:
            self.is_loading = True
            self.error_message = ""
            self.df_data = []
            self.df_columns = []
        try:
            # 1) Upload to volume
            hp = self.http_path or HTTP_PATH
            vol = self.volume_path.strip() or f"/Volumes/{UPLOAD_VOLUME.replace('.', '/')}"
            if not vol.startswith("/Volumes/"):
                parts = UPLOAD_VOLUME.split(".")
                vol = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
            w = get_workspace_client()
            file_bytes = await file.read()
            binary_data = io.BytesIO(file_bytes)
            path = f"{vol.rstrip('/')}/{file.filename}"
            w.files.upload(path, binary_data, overwrite=True)

            # 2) Run ai_parse_document via SQL
            conn = get_connection(hp)
            ensure_results_table(conn, self.table_name or RESULTS_TABLE)
            parsed = run_parse_document_sql(conn, vol, file.filename)
            if not parsed:
                async with self:
                    self.error_message = "No parsed document returned from ai_parse_document."
                    self.is_loading = False
                return
            text = extract_text_from_parsed(parsed)
            if not text.strip():
                async with self:
                    self.error_message = "Parsed document had no text content."
                    self.is_loading = False
                return

            # 3) Invoke agent
            agent_out = invoke_extraction_agent(
                text, self.agent_endpoint or AGENT_ENDPOINT
            )
            ingest_id = str(uuid.uuid4())
            today = date.today().isoformat()
            exploded = explode_agent_output(
                agent_out, today, file.filename, ingest_id
            )

            # 4) Append to results table
            insert_rows(conn, self.table_name or RESULTS_TABLE, exploded)

            # 5) Load new rows into editor
            where = f"ingest_id = '{ingest_id.replace(chr(39), chr(39)+chr(39))}'"
            df = read_table(conn, self.table_name or RESULTS_TABLE, where)
            data, cols = pandas_to_editor_format(df)
            async with self:
                self.uploaded_filename = file.filename
                self.ingest_id = ingest_id
                self.df_data = data
                self.df_columns = cols
                self.error_message = ""
        except Exception as e:
            logger.exception("run_ingest: %s", e)
            async with self:
                self.error_message = str(e)
        finally:
            async with self:
                self.is_loading = False

    @rx.event(background=True)
    async def save_ingest(self):
        async with self:
            self.is_saving = True
            self.error_message = ""
        try:
            if not self.df_data or not self.df_columns:
                async with self:
                    self.error_message = "No data to save."
                    self.is_saving = False
                return
            col_names = [c["title"] for c in self.df_columns]
            df = pd.DataFrame(self.df_data, columns=col_names)
            conn = get_connection(self.http_path or HTTP_PATH)
            update_rows_by_id(conn, self.table_name or RESULTS_TABLE, df)
            yield rx.toast("Saved successfully.", level="success")
        except Exception as e:
            logger.exception("save_ingest: %s", e)
            yield rx.toast(f"Error saving: {e}", level="error")
        finally:
            async with self:
                self.is_saving = False


# ---------------------------------------------------------------------------
# State: Explore page
# ---------------------------------------------------------------------------
class ExploreState(rx.State):
    http_path: str = ""
    table_name: str = RESULTS_TABLE

    filter_manufacturer: str = ""
    filter_ingest_date: str = ""
    filter_cylinder_count: str = ""
    filter_vermeer_product: str = ""

    df_data: list[list[Any]] = []
    df_columns: list[dict] = []
    is_loading: bool = False
    is_saving: bool = False
    error_message: str = ""

    @rx.var
    def columns_for_editor(self) -> list[dict]:
        return self.df_columns

    @rx.var
    def data_for_editor(self) -> list[list[Any]]:
        return self.df_data

    @rx.event
    def handle_cell_change(self, cell: tuple[int, int], new_value: Any):
        # on_cell_edited passes (cell, new_value); cell is (col_index, row_index)
        col_index, row_index = cell[0], cell[1]
        val = getattr(new_value, "data", new_value)
        if 0 <= row_index < len(self.df_data) and 0 <= col_index < len(self.df_data[row_index]):
            self.df_data[row_index][col_index] = val

    def _build_where(self) -> str:
        parts = []
        if self.filter_manufacturer.strip():
            m = self.filter_manufacturer.strip().replace("'", "''")
            parts.append(f"company = '{m}'")
        if self.filter_ingest_date.strip():
            parts.append(f"ingest_date = '{self.filter_ingest_date.strip()}'")
        if self.filter_cylinder_count.strip():
            parts.append(f"number_of_cylinders = {self.filter_cylinder_count.strip()}")
        if self.filter_vermeer_product.strip():
            v = self.filter_vermeer_product.strip().replace("'", "''")
            parts.append(f"vermeer_product = '{v}'")
        return " AND ".join(parts) if parts else "1=1"

    @rx.event(background=True)
    async def load_filtered(self):
        async with self:
            self.is_loading = True
            self.error_message = ""
            self.df_data = []
            self.df_columns = []
        try:
            conn = get_connection(self.http_path or HTTP_PATH)
            where = self._build_where()
            df = read_table(conn, self.table_name or RESULTS_TABLE, where)
            data, cols = pandas_to_editor_format(df)
            async with self:
                self.df_data = data
                self.df_columns = cols
        except Exception as e:
            logger.exception("load_filtered: %s", e)
            async with self:
                self.error_message = str(e)
        finally:
            async with self:
                self.is_loading = False

    @rx.event(background=True)
    async def save_explore(self):
        async with self:
            self.is_saving = True
            self.error_message = ""
        try:
            if not self.df_data or not self.df_columns:
                async with self:
                    self.error_message = "No data to save."
                    self.is_saving = False
                return
            col_names = [c["title"] for c in self.df_columns]
            df = pd.DataFrame(self.df_data, columns=col_names)
            conn = get_connection(self.http_path or HTTP_PATH)
            update_rows_by_id(conn, self.table_name or RESULTS_TABLE, df)
            yield rx.toast("Saved successfully.", level="success")
        except Exception as e:
            logger.exception("save_explore: %s", e)
            yield rx.toast(f"Error saving: {e}", level="error")
        finally:
            async with self:
                self.is_saving = False


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def ingest_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading("Engine spec ingest", size="8"),
            rx.text("Upload a PDF to parse, extract engine data, and append to the results table."),
            rx.input(
                placeholder="SQL warehouse HTTP path",
                value=IngestState.http_path,
                on_change=IngestState.set_http_path,
                width="100%",
            ),
            rx.input(
                placeholder="Results table (catalog.schema.table)",
                value=IngestState.table_name,
                on_change=IngestState.set_table_name,
                width="100%",
            ),
            rx.input(
                placeholder="Volume path (e.g. /Volumes/main/default/uploads)",
                value=IngestState.volume_path,
                on_change=IngestState.set_volume_path,
                width="100%",
            ),
            rx.input(
                placeholder="Agent endpoint name",
                value=IngestState.agent_endpoint,
                on_change=IngestState.set_agent_endpoint,
                width="100%",
            ),
            rx.upload(
                rx.vstack(
                    rx.button("Select PDF", color_scheme="blue"),
                    rx.text(IngestState.uploaded_filename),
                ),
                id="ingest_upload",
                max_files=1,
                accept={"application/pdf": [".pdf"]},
            ),
            rx.cond(
                IngestState.is_loading,
                rx.spinner(),
                rx.button(
                    "Parse & ingest",
                    on_click=IngestState.run_ingest(rx.upload_files(upload_id="ingest_upload")),
                    color_scheme="green",
                ),
            ),
            rx.cond(
                IngestState.error_message != "",
                rx.callout(IngestState.error_message, icon="triangle_alert", color_scheme="red"),
                rx.fragment(),
            ),
            rx.divider(),
            rx.cond(
                IngestState.df_columns,
                rx.vstack(
                    rx.text("Edit rows and click Save to finalize."),
                    rx.data_editor(
                        data=IngestState.data_for_editor,
                        columns=IngestState.columns_for_editor,
                        on_cell_edited=IngestState.handle_cell_change,
                        row_height=40,
                    ),
                    rx.button(
                        "Save",
                        on_click=IngestState.save_ingest,
                        loading=IngestState.is_saving,
                        color_scheme="blue",
                    ),
                    width="100%",
                    align_items="stretch",
                ),
                rx.text("No data yet. Upload a PDF and run Parse & ingest."),
            ),
            width="100%",
            spacing="4",
            align_items="stretch",
        ),
        padding="6",
    )


def explore_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading("Explore engine specs", size="8"),
            rx.text("Filter and load data, then view, edit, and save."),
            rx.input(
                placeholder="SQL warehouse HTTP path",
                value=ExploreState.http_path,
                on_change=ExploreState.set_http_path,
                width="100%",
            ),
            rx.input(
                placeholder="Results table (catalog.schema.table)",
                value=ExploreState.table_name,
                on_change=ExploreState.set_table_name,
                width="100%",
            ),
            rx.grid(
                rx.vstack(
                    rx.text("Manufacturer"),
                    rx.input(
                        placeholder="e.g. DEUTZ",
                        value=ExploreState.filter_manufacturer,
                        on_change=ExploreState.set_filter_manufacturer,
                    ),
                ),
                rx.vstack(
                    rx.text("Ingest date"),
                    rx.input(
                        placeholder="YYYY-MM-DD",
                        value=ExploreState.filter_ingest_date,
                        on_change=ExploreState.set_filter_ingest_date,
                    ),
                ),
                rx.vstack(
                    rx.text("Cylinder count"),
                    rx.input(
                        placeholder="e.g. 4",
                        value=ExploreState.filter_cylinder_count,
                        on_change=ExploreState.set_filter_cylinder_count,
                    ),
                ),
                rx.vstack(
                    rx.text("Vermeer product"),
                    rx.input(
                        placeholder="Filter by product",
                        value=ExploreState.filter_vermeer_product,
                        on_change=ExploreState.set_filter_vermeer_product,
                    ),
                ),
                columns="4",
                width="100%",
            ),
            rx.cond(
                ExploreState.is_loading,
                rx.spinner(),
                rx.button("Load data", on_click=ExploreState.load_filtered, color_scheme="green"),
            ),
            rx.cond(
                ExploreState.error_message != "",
                rx.callout(ExploreState.error_message, icon="triangle_alert", color_scheme="red"),
                rx.fragment(),
            ),
            rx.divider(),
            rx.cond(
                ExploreState.df_columns,
                rx.vstack(
                    rx.text("Edit rows and click Save to update the table."),
                    rx.data_editor(
                        data=ExploreState.data_for_editor,
                        columns=ExploreState.columns_for_editor,
                        on_cell_edited=ExploreState.handle_cell_change,
                        row_height=40,
                    ),
                    rx.button(
                        "Save",
                        on_click=ExploreState.save_explore,
                        loading=ExploreState.is_saving,
                        color_scheme="blue",
                    ),
                    width="100%",
                    align_items="stretch",
                ),
                rx.text("Set filters and click Load data."),
            ),
            width="100%",
            spacing="4",
            align_items="stretch",
        ),
        padding="6",
    )


def layout_with_nav(children: rx.Component) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.link(rx.button("Ingest"), href="/", padding="2"),
            rx.link(rx.button("Explore"), href="/explore", padding="2"),
            spacing="4",
            padding="4",
            border_bottom="1px solid",
        ),
        children,
        width="100%",
    )


def index() -> rx.Component:
    return layout_with_nav(ingest_page())


def explore() -> rx.Component:
    return layout_with_nav(explore_page())


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = rx.App()
app.add_page(index, route="/", title="Engine spec ingest")
app.add_page(explore, route="/explore", title="Explore engine specs")
