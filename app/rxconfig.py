"""Reflex config when running from app/ (e.g. Databricks Apps)."""
import reflex as rx

# Use engine_spec so a root-level app.py (if present) does not shadow the package.
# state_auto_setters=True keeps auto-generated setters (e.g. set_http_path) until 0.9.
config = rx.Config(app_name="engine_spec", state_auto_setters=True)
