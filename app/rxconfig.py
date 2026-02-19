"""Reflex config when running from app/ (e.g. Databricks Apps)."""
import reflex as rx

# Use engine_spec so a root-level app.py (if present) does not shadow the package.
config = rx.Config(app_name="engine_spec")
