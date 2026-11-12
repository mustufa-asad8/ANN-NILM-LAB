"""Streamlit Cloud entrypoint for the NILM dashboard."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "src" / "nilm_ccp" / "dashboard.py"),
    run_name="__main__",
)
