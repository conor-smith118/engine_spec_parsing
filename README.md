# Engine Spec Parsing

A Databricks Dash app for ingesting engine specification PDFs, extracting structured data via AI, and exploring or editing the results. Built for Databricks Apps (Python apps running in the workspace).

## What the app does

The app turns engine spec PDFs into structured data you can query and edit. You upload PDFs, the app parses them with Databricks `ai_parse_document`, runs an Information Extraction Agent to pull out fields (manufacturer, cylinder count, power ratings, etc.), and writes the results to a Delta table. From there you can filter, explore, and correct the data.

**Ingest** – Upload PDFs, watch progress as each file is parsed and extracted, then review and save rows to your results table. Uses background callbacks so the UI stays responsive.

**Explore** – Filter by manufacturer, ingest date, cylinder count, or product. View an embedded dashboard and chat with a Knowledge Assistant that answers questions about the specs using RAG over your data.

**Edit Data** – Load filtered rows into an editable table, fix values, and save changes back to Delta.

**Admin** – Configure SQL warehouse, results table, volume path, agent endpoints, dashboard URL, and Knowledge Assistant endpoint. Settings are stored in the browser and used across the app, so you can adjust them after deployment without changing code.

## Project structure

- **`app/`** – Dash app source
  - `app.py` – main application
  - `app.yaml` – Databricks App run config
  - `requirements.txt` – Python dependencies
- **`resources/`** – Databricks resource configs (jobs, pipelines)
- **`tests/`** – Unit tests
- **`.github/workflows/`** – CI/CD (deploy app to workspace)

## Setup for other workspaces

To run this app in a different Databricks workspace or account:

### 1. Prerequisites

- Databricks workspace with Unity Catalog
- SQL warehouse
- Service principal or user with appropriate permissions (see below)
- (Optional) Information Extraction Agent endpoint for `ai_query`
- (Optional) Knowledge Assistant for Explore chat
- (Optional) Published dashboard for Explore embed

### 2. Edit defaults in `app/app.py`

Update the constants near the top of `app/app.py` to match your workspace:

| Constant | Description | Example |
|----------|-------------|---------|
| `DATABRICKS_HOST` | Workspace URL | `https://your-workspace.cloud.databricks.com` |
| `HTTP_PATH` | SQL warehouse HTTP path | `/sql/1.0/warehouses/<warehouse-id>` |
| `RESULTS_TABLE` | Delta table for parsed data | `catalog.schema.parsed_engine_data` |
| `UPLOAD_VOLUME` | Unity Catalog volume for uploads | `catalog.schema.app_storage` |
| `AGENT_ENDPOINT` | Information Extraction Agent endpoint name | `kie-xxxx-endpoint` |
| `EXPLORE_KA_ENDPOINT` | Knowledge Assistant endpoint (Explore chat) | `ka-xxxx-endpoint` |
| `EXPLORE_DASHBOARD_URL` | Published dashboard embed URL | `https://.../embed/dashboardsv3/...` |
| `EXTRACTION_INPUT_TABLE` | One-row table for `ai_query` input | `catalog.schema._extraction_input` |
| `KNOWLEDGE_ASSISTANT_ID` | Knowledge Assistant ID (for sync) | UUID from KA settings |
| `VOLUME_GRANTEE_DEFAULT` | Principal ID with `READ_VOLUME` on volumes | Service principal object ID |

The results table and extraction input table are created automatically on first ingest if they do not exist.

### 3. Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABRICKS_HOST` | No* | Workspace URL (overrides `DATABRICKS_HOST` constant) |
| `DATABRICKS_TOKEN` | Yes** | Personal access token or OAuth token |
| `AGENT_ENDPOINT` | No | Information Extraction Agent endpoint (overrides default) |
| `KA_TOKEN` | No | Token for Knowledge Assistant sync (or use `dbutils.secrets` scope `css_tokens`, key `ka_token`) |
| `VOLUME_GRANTEE` | No | Principal ID for volume listing (default: `VOLUME_GRANTEE_DEFAULT`) |
| `DATABRICKS_APP_PORT` / `PORT` | No | App port (default: 8080) |
| `DASH_CACHE_DIR` | No | Disk cache dir for background callbacks (default: `/tmp/dash_cache`) |

\* When running as a Databricks App, host and auth are usually provided by the platform.  
\** For local runs or CI, `DATABRICKS_TOKEN` is required. For Databricks Apps, OAuth or service principal is typically used.

### 4. Permissions (app identity)

The app identity (user or service principal) needs:

- **SQL warehouse**: `CAN USE`
- **Results table**: `SELECT`, `INSERT`, `UPDATE`; table created on first ingest if missing
- **Upload volume**: `READ VOLUME`, `WRITE VOLUME`; `USE` on catalog and schema
- **Information Extraction Agent endpoint**: `CAN QUERY`
- **Knowledge Assistant endpoint** (if used): `CAN QUERY`

### 5. Run locally

```bash
cd app
pip install -r requirements.txt
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
python app.py
```

Then open `http://localhost:8080` (or the port shown).

### 6. Deploy as a Databricks App

**Option A: GitHub Actions**

1. Add these secrets to your repo:
   - `DATABRICKS_HOST` – workspace URL
   - `DATABRICKS_TOKEN` – PAT or OAuth token
   - `DATABRICKS_USER_EMAIL` – user email for workspace path
   - `DATABRICKS_APP_DEV_NAME` – app name for `dev` branch
   - `DATABRICKS_APP_PROD_NAME` – app name for `main` branch

2. Push to `dev` or `main`; the workflow syncs `app/` and deploys the app.

**Option B: Manual deploy**

```bash
databricks sync ./app /Workspace/Users/<your-email>/apps/engine_spec_parsing
databricks apps deploy engine_spec_parsing --source-code-path /Workspace/Users/<your-email>/apps/engine_spec_parsing
```

### 7. Admin page configuration

After deployment, use the **Admin** page to adjust settings without changing code:

- SQL warehouse HTTP path
- Results table
- Volume path
- Information Extraction Agent endpoint name
- Explore dashboard URL
- Knowledge Assistant endpoint name

These are stored in the browser and override the defaults.

## Getting started (development)

- **In Databricks workspace**: [Databricks Bundles – workspace](https://docs.databricks.com/dev-tools/bundles/workspace)
- **Locally with IDE**: [Databricks VS Code extension](https://docs.databricks.com/dev-tools/vscode-ext.html)
- **CLI**: [Databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html)

Install dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```
