# Engine Spec Parsing – Databricks Reflex App

Reflex app with two pages, following the [Databricks Apps Cookbook (Reflex)](https://apps-cookbook.dev/docs/category/reflex).

## Pages

1. **Ingest** (`/`)
   - Upload a PDF to a Unity Catalog volume.
   - Run `ai_parse_document` on the file (via SQL warehouse).
   - Call the Information Extraction Agent at `kie-c2e65325-endpoint` (or configured endpoint).
   - Explode the agent JSON (company, product_series, engines) into one row per engine.
   - Append rows to the results Delta table.
   - Show the new rows in an editable table; edit and click **Save** to finalize.

2. **Explore** (`/explore`)
   - Set filters: Manufacturer, Ingest Date, Cylinder count, Vermeer Product.
   - Click **Load data** to query the results table.
   - View and edit rows in the table; click **Save** to update the corresponding rows by `id`.

## Environment variables (optional)

Set in the Databricks App configuration or in the UI (first run you can type them in the app):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABRICKS_HTTP_PATH` | SQL warehouse HTTP path | (required; can be entered in UI) |
| `RESULTS_TABLE` | Full table name `catalog.schema.table` | `main.default.engine_spec_results` |
| `UPLOAD_VOLUME` | Volume for PDF uploads `catalog.schema.volume` | `main.default.uploads` |
| `AGENT_ENDPOINT` | Model serving endpoint for extraction | `kie-c2e65325-endpoint` |

## Permissions

- **SQL warehouse**: `CAN USE`.
- **Results table**: `SELECT`, `MODIFY` (create if not exists, insert, update).
- **Upload volume**: `READ VOLUME`, `WRITE VOLUME`; schema/catalog `USE SCHEMA` / `USE CATALOG`.
- **Serving endpoint**: `CAN QUERY` on the Information Extraction Agent endpoint.

## Results table schema

The app creates the table if it does not exist:

- `id` (STRING) – UUID per row, used for updates.
- `company`, `product_series`, `engine_type`, `power_rating_continuous_operations`, `number_of_cylinders`, `specific_fuel_consumption`, `ingest_date`, `vermeer_product`, `source_file`, `ingest_id` (STRING).

## Project structure (for Reflex)

Reflex is configured with `app_name="engine_spec"` so the app lives in a package that won’t conflict with a root-level `app.py` if one exists in the deployment. Layout:

- `app/` – Databricks app root (deployed as `source_code/`)
  - `rxconfig.py`, `app.yaml`, `requirements.txt`
  - `engine_spec/` – Reflex package
    - `__init__.py`, `engine_spec.py` – app code and `rx.App()`

The Reflex module is `engine_spec.engine_spec`. Do not add an `app.py` at the root of this folder, or remove it if present, so the correct package is used.

## Running locally

From the Databricks app directory (the one that contains `rxconfig.py` and the inner `app/` folder):

```bash
cd app
pip install -r requirements.txt
reflex run
```

Then open the URL shown (e.g. http://localhost:3000). Enter the SQL warehouse HTTP path and table/volume/endpoint as needed.
