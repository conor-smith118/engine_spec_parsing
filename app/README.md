# Engine Spec Parsing (Dash)

Dash app for Databricks Apps: ingest PDFs (parse + agent → Delta) and explore/edit results.

## Hardcoded config

- **Host**: `https://adb-878896594214094.14.azuredatabricks.net` (override with `DATABRICKS_HOST` if needed)
- **SQL warehouse HTTP path**: `/sql/1.0/warehouses/2e35be694d7a3467`
- **Results table**: `conor_smith.engine_specs_parse.parsed_engine_data` (created on first ingest)
- **Extraction input table**: `conor_smith.engine_specs_parse._extraction_input` (one-row table for `ai_query`; created and cleared by the app)
- **Upload volume**: `conor_smith.engine_specs_parse.app_storage` → `/Volumes/conor_smith/engine_specs_parse/app_storage`

Optional env override: `AGENT_ENDPOINT` (default `kie-c2e65325-endpoint`). `DATABRICKS_APP_PORT` or `PORT` for the app port (Databricks sets this).

## Permissions (app service principal)

- **SQL warehouse**: `CAN USE`
- **Results table**: `SELECT`, `INSERT`, `UPDATE`; table will be created if missing
- **Upload volume**: `READ VOLUME`, `WRITE VOLUME`; schema/catalog `USE`
- **Serving endpoint**: `CAN QUERY` on the agent endpoint

## Run locally

```bash
cd app
pip install -r requirements.txt
export DATABRICKS_HOST=... DATABRICKS_TOKEN=...  # or use profile
python app.py
```

Then open `http://localhost:8080` (or the port shown).
