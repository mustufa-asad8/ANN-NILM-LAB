"""Streamlit dashboard for the ANN-based NILM project."""

from __future__ import annotations

import io
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "nilm_ccp" / "train.py"
OUTPUT_DIR = ROOT / "artifacts"
REGION_FILES = {
    "New York": ROOT / "data" / "raw" / "new_york" / "readings_1min.csv",
    "California": ROOT / "data" / "raw" / "california" / "readings_1min.csv",
    "Austin": ROOT / "data" / "raw" / "austin" / "readings_1min.csv",
}
HOSTED_REGION_FILES = {
    "Austin": ROOT / "data" / "hosted" / "austin_eligible_homes.csv.gz",
}
LEGACY_REGION_FILES = {
    "New York": ROOT / "1minute_data_newyork" / "1minute_data_newyork.csv",
    "California": ROOT
    / "1minute_data_california"
    / "1minute_data_california"
    / "1minute_data_california.csv",
    "Austin": ROOT
    / "1minute_data_austin"
    / "1minute_data_austin"
    / "1minute_data_austin.csv",
}
REGION_KEYS = {"New York": "newyork", "California": "california", "Austin": "austin"}
REQUIRED_SIGNAL_COLUMNS = ("grid", "air1", "furnace1", "drye1")
MIN_COMPLETE_READINGS = 10_000
MIN_CLASS_READINGS = 100
DEFAULT_HOME_COUNT = 5
PLOT_FILES = {
    "chronological_test": (
        "chronological_test_confusion_matrices.png",
        "chronological_test_power_predictions.png",
    ),
    "unseen_home": (
        "unseen_home_confusion_matrices.png",
        "unseen_home_power_predictions.png",
    ),
}


st.set_page_config(
    page_title="NILM Lab | ANN Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#14231f; --muted:#60716b; --paper:#f5f7f3; --green:#23745b; --lime:#c8ed74; }
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stSidebar"] { background:#e9efe9; border-right:1px solid #d8e1da; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    .hero {
      padding:2rem 2.2rem; border-radius:22px; color:#f6faf3;
      background:radial-gradient(circle at 82% 10%, #398367 0, transparent 30%),
                 linear-gradient(120deg,#132c26 0%,#1c493a 68%,#235b45 100%);
      margin:0 0 1.4rem; position:relative; overflow:hidden;
    }
    .hero h1 { font-size:2.35rem; letter-spacing:-.06em; line-height:1.08; margin:.45rem 0 .65rem; }
    .hero p { color:#c5d8ca; max-width:750px; margin:0; font-size:1rem; }
    .eyebrow { font:500 .72rem Consolas,monospace; color:#c8ed74; letter-spacing:.14em; text-transform:uppercase; }
    .section-label { color:#60716b; font:500 .72rem Consolas,monospace; letter-spacing:.1em; text-transform:uppercase; }
    .mini-card {
      background:#fff; border:1px solid #e1e8e1; border-radius:16px;
      padding:1rem 1.15rem; min-height:94px;
    }
    .mini-card .value { color:#18382d; font-size:1.35rem; font-weight:800; letter-spacing:-.04em; }
    .mini-card .label { color:#72817b; font-size:.78rem; margin-top:.15rem; }
    .note { border-left:3px solid #98c65f; padding:.45rem .8rem; color:#60716b; font-size:.83rem; }
    div[data-testid="stMetric"] {
      background:#fff; border:1px solid #e1e8e1; border-radius:14px; padding:.8rem 1rem;
    }
    div.stButton > button[kind="primary"] {
      background:#23745b; border:0; border-radius:10px; font-weight:700; min-height:2.8rem;
    }
    div.stButton > button[kind="primary"]:hover { background:#195941; border:0; }
    .mono { font-family:Consolas,monospace; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_home_ids(value: str) -> list[int]:
    try:
        homes = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError("Enter home IDs as comma-separated whole numbers.") from exc
    if len(homes) < 5:
        raise ValueError("The CCP requires at least five homes.")
    if len(set(homes)) != len(homes):
        raise ValueError("Home IDs must not repeat.")
    return homes


@st.cache_data(show_spinner="Scanning this region for homes with usable readings...")
def find_eligible_homes(
    csv_path: str, file_size: int, modified_time_ns: int
) -> list[int]:
    del file_size, modified_time_ns
    columns = ["dataid", *REQUIRED_SIGNAL_COLUMNS]
    dtypes = {
        "dataid": "int32",
        **{column: "float32" for column in REQUIRED_SIGNAL_COLUMNS},
    }
    counts: dict[int, list[int]] = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=columns,
        dtype=dtypes,
        chunksize=500_000,
        low_memory=False,
    ):
        usable = chunk.dropna(subset=columns)
        if usable.empty:
            continue
        group_ids, home_ids = pd.factorize(usable["dataid"], sort=False)
        group_count = len(home_ids)
        complete_counts = np.bincount(group_ids, minlength=group_count)
        appliance_counts = []
        for appliance in ("air1", "furnace1", "drye1"):
            is_on = usable[appliance].to_numpy() >= 0.05
            appliance_counts.append(
                (
                    np.bincount(
                        group_ids, weights=is_on, minlength=group_count
                    ).astype(np.int64),
                    np.bincount(
                        group_ids, weights=~is_on, minlength=group_count
                    ).astype(np.int64),
                )
            )
        for index, home_id in enumerate(home_ids):
            home_counts = counts.setdefault(int(home_id), [0] * 7)
            home_counts[0] += int(complete_counts[index])
            for appliance_index, (on_counts, off_counts) in enumerate(
                appliance_counts
            ):
                home_counts[appliance_index + 1] += int(on_counts[index])
                home_counts[appliance_index + 4] += int(off_counts[index])

    return sorted(
        home_id
        for home_id, home_counts in counts.items()
        if home_counts[0] >= MIN_COMPLETE_READINGS
        and all(
            home_counts[index] >= MIN_CLASS_READINGS
            for index in range(1, 7)
        )
    )


def random_home_selection(
    eligible_homes: list[int], home_count: int
) -> tuple[list[int], int]:
    rng = random.SystemRandom()
    selected = rng.sample(eligible_homes, home_count)
    test_home = rng.choice(selected)
    return selected, test_home


def load_saved_results() -> tuple[pd.DataFrame | None, dict | None]:
    metrics_path = OUTPUT_DIR / "metrics" / "metrics.csv"
    metadata_path = OUTPUT_DIR / "metadata" / "run_metadata.json"
    if not metrics_path.is_file():
        return None, None
    metrics = pd.read_csv(metrics_path)
    metadata = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metrics, metadata


def build_submission_bundle() -> bytes:
    checkpoint_files = [
        OUTPUT_DIR / "checkpoints" / checkpoint_name
        for checkpoint_name in (
            "mlp_best.pt",
            "cnn_1d_best.pt",
            "lstm_best.pt",
            "gru_best.pt",
        )
    ]
    files = [
        SCRIPT,
        ROOT / "app.py",
        ROOT / "src" / "nilm_ccp" / "dashboard.py",
        ROOT / "src" / "nilm_ccp" / "__init__.py",
        ROOT / "README.md",
        ROOT / "data" / "README.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-training.txt",
        ROOT / "Dockerfile",
        ROOT / "compose.yaml",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "tests" / "test_training.py",
        ROOT / "data" / "hosted" / "austin_eligible_homes.csv.gz",
        OUTPUT_DIR / "reports" / "report.md",
        OUTPUT_DIR / "metrics" / "metrics.csv",
        OUTPUT_DIR / "metadata" / "run_metadata.json",
        OUTPUT_DIR / "logs" / "training_log.csv",
        *[path for path in checkpoint_files if path.is_file()],
        *[
            OUTPUT_DIR / "figures" / filename
            for figure_names in PLOT_FILES.values()
            for filename in figure_names
        ],
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot create the submission bundle; required output files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT))
    return bundle.getvalue()


def run_training(command: list[str], log_placeholder, progress_bar) -> tuple[int, str]:
    lines: queue.Queue[str | None] = queue.Queue()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Could not read the training process output.")

    def read_output() -> None:
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output: list[str] = []
    finished_reading = False
    epoch_pattern = re.compile(r"epoch\s+(\d+)/(\d+)", re.IGNORECASE)
    while process.poll() is None or not finished_reading:
        try:
            line = lines.get(timeout=0.2)
            if line is None:
                finished_reading = True
            else:
                output.append(line)
                match = epoch_pattern.search(line)
                if match:
                    progress_bar.progress(
                        min(0.98, int(match.group(1)) / max(1, int(match.group(2)))),
                        text=f"Training models · epoch {match.group(1)} of {match.group(2)}",
                    )
                log_placeholder.code("\n".join(output[-28:]), language="text")
        except queue.Empty:
            if process.poll() is None:
                time.sleep(0.1)
    reader.join(timeout=1)
    return_code = process.wait()
    return return_code, "\n".join(output)


def metric_cards(metrics: pd.DataFrame, test_group: str) -> None:
    subset = metrics[metrics["test_group"] == test_group]
    if subset.empty:
        st.info("No results for this test split were found.")
        return
    best_f1 = subset.loc[subset["macro_f1"].idxmax()]
    best_balanced = subset.loc[subset["balanced_accuracy"].idxmax()]
    mean_mae = subset["mae_w"].mean()
    first, second, third = st.columns(3)
    first.metric(
        "Best macro-F1",
        f"{best_f1['macro_f1']:.3f}",
        f"{best_f1['model']} · {best_f1['appliance']}",
    )
    second.metric(
        "Best balanced accuracy",
        f"{best_balanced['balanced_accuracy']:.3f}",
        f"{best_balanced['model']} · {best_balanced['appliance']}",
    )
    third.metric("Mean power MAE", f"{mean_mae:.1f} W", "Across models and appliances")


def results_table(metrics: pd.DataFrame, test_group: str) -> pd.DataFrame:
    columns = [
        "model",
        "appliance",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "status_probability_threshold",
        "mae_w",
        "rmse_w",
    ]
    table = metrics.loc[metrics["test_group"] == test_group, columns].copy()
    table["Accuracy target"] = table["accuracy"].map(
        lambda value: "PASS" if value >= 0.85 else "FAIL"
    )
    table["Balanced accuracy target"] = table["balanced_accuracy"].map(
        lambda value: "PASS" if value >= 0.75 else "FAIL"
    )
    table["Macro-F1 target"] = table["macro_f1"].map(
        lambda value: "PASS" if value >= 0.70 else "FAIL"
    )
    return table


def render_results(metrics: pd.DataFrame, metadata: dict | None) -> None:
    required_models = {"MLP", "1D-CNN", "LSTM", "GRU"}
    saved_models = set(metrics["model"].dropna().astype(str))
    if not required_models.issubset(saved_models):
        st.info(
            "These are results from an earlier run. Train all four models to "
            "refresh the comparison with MLP, 1D-CNN, LSTM, and GRU."
        )
    if metadata:
        home_ids = ", ".join(str(home) for home in metadata.get("home_ids", []))
        st.caption(
            f"Latest run · {metadata.get('region', 'unknown').replace('newyork', 'New York').replace('california', 'California').replace('austin', 'Austin')} · "
            f"Homes {home_ids} · seed {metadata.get('seed', '—')}"
        )
    tab_chrono, tab_unseen, tab_training, tab_files = st.tabs(
        ["Chronological test", "Unseen home", "Training history", "Files & details"]
    )
    with tab_chrono:
        st.markdown('<div class="section-label">Later-time holdout</div>', unsafe_allow_html=True)
        metric_cards(metrics, "chronological_test")
        st.dataframe(
            results_table(metrics, "chronological_test"),
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.NumberColumn(format="%.3f"),
                "balanced_accuracy": st.column_config.NumberColumn(format="%.3f"),
                "macro_f1": st.column_config.NumberColumn(format="%.3f"),
                "status_probability_threshold": st.column_config.NumberColumn(
                    "Status cutoff", format="%.3f"
                ),
                "mae_w": st.column_config.NumberColumn("MAE (W)", format="%.1f"),
                "rmse_w": st.column_config.NumberColumn("RMSE (W)", format="%.1f"),
            },
        )
        st.caption(
            "CCP minimums: accuracy 0.85 · balanced accuracy 0.75 · macro-F1 0.70. "
            "Status cutoffs are selected on validation data only."
        )
        render_plots("chronological_test")
    with tab_unseen:
        st.markdown('<div class="section-label">Held-out household</div>', unsafe_allow_html=True)
        metric_cards(metrics, "unseen_home")
        st.dataframe(
            results_table(metrics, "unseen_home"),
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.NumberColumn(format="%.3f"),
                "balanced_accuracy": st.column_config.NumberColumn(format="%.3f"),
                "macro_f1": st.column_config.NumberColumn(format="%.3f"),
                "status_probability_threshold": st.column_config.NumberColumn(
                    "Status cutoff", format="%.3f"
                ),
                "mae_w": st.column_config.NumberColumn("MAE (W)", format="%.1f"),
                "rmse_w": st.column_config.NumberColumn("RMSE (W)", format="%.1f"),
            },
        )
        st.caption(
            "CCP minimums: accuracy 0.85 · balanced accuracy 0.75 · macro-F1 0.70. "
            "Status cutoffs are selected on validation data only."
        )
        render_plots("unseen_home")
    with tab_training:
        training_path = OUTPUT_DIR / "logs" / "training_log.csv"
        if training_path.is_file():
            history = pd.read_csv(training_path)
            if not history.empty:
                st.subheader("Validation loss by epoch")
                figure, axis = plt.subplots(figsize=(9, 3.5))
                for model_name, model_history in history.groupby("model"):
                    axis.plot(
                        model_history["epoch"],
                        model_history["validation_loss"],
                        marker="o",
                        markersize=4,
                        linewidth=2,
                        label=model_name,
                    )
                axis.set_xlabel("Epoch")
                axis.set_ylabel("Validation loss")
                axis.grid(alpha=0.25)
                axis.legend(frameon=False)
                figure.tight_layout()
                st.pyplot(figure, width="stretch")
                plt.close(figure)
                st.dataframe(history, width="stretch", hide_index=True)
        else:
            st.info("Train a model to populate the training history.")
    with tab_files:
        if metadata:
            st.subheader("Run setup")
            st.json(metadata, expanded=False)
        report_path = OUTPUT_DIR / "reports" / "report.md"
        if report_path.is_file():
            st.download_button(
                "Download project report",
                report_path.read_bytes(),
                file_name="NILM_CCP_REPORT.md",
                mime="text/markdown",
            )
        st.download_button(
            "Download submission bundle (.zip)",
            build_submission_bundle(),
            file_name="nilm_ccp_submission.zip",
            mime="application/zip",
            width="stretch",
        )
        metrics_path = OUTPUT_DIR / "metrics" / "metrics.csv"
        if metrics_path.is_file():
            st.download_button(
                "Download metrics CSV",
                metrics_path.read_bytes(),
                file_name="nilm_metrics.csv",
                mime="text/csv",
            )


def render_plots(test_group: str) -> None:
    confusion_name, power_name = PLOT_FILES[test_group]
    first, second = st.columns(2)
    confusion = OUTPUT_DIR / "figures" / confusion_name
    power = OUTPUT_DIR / "figures" / power_name
    with first:
        if confusion.is_file():
            st.image(str(confusion), caption="ON/OFF confusion matrices", width="stretch")
        else:
            st.info("Confusion matrix will appear after evaluation.")
    with second:
        if power.is_file():
            st.image(str(power), caption="Actual and predicted appliance power", width="stretch")
        else:
            st.info("Power chart will appear after evaluation.")


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">NILM LAB · CT-466</div>
          <h1>See what the meter<br>can't show.</h1>
          <p>Train and compare two neural networks that disaggregate household
          power into appliance-level status and consumption.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚡ NILM Lab")
        st.caption("ANN-based non-intrusive load monitoring")
        st.divider()
        st.markdown('<div class="section-label">Experiment setup</div>', unsafe_allow_html=True)
        region_name = st.selectbox(
            "Dataset region",
            list(REGION_FILES),
            index=2,
            key="dataset_region",
        )
        default_dataset = REGION_FILES[region_name]
        if not default_dataset.is_file():
            default_dataset = HOSTED_REGION_FILES.get(region_name, default_dataset)
        dataset_path_key = f"dataset_path_{REGION_KEYS[region_name]}"
        saved_dataset_path = st.session_state.get(dataset_path_key)
        if (
            saved_dataset_path is not None
            and (
                Path(str(saved_dataset_path)).expanduser().resolve()
                == LEGACY_REGION_FILES[region_name].resolve()
                or (
                    not REGION_FILES[region_name].is_file()
                    and Path(str(saved_dataset_path)).expanduser().resolve()
                    == REGION_FILES[region_name].resolve()
                )
            )
        ):
            st.session_state[dataset_path_key] = str(default_dataset)
        dataset_path = st.text_input(
            "CSV file path",
            value=str(default_dataset),
            key=dataset_path_key,
        )
        selected_file = Path(dataset_path).expanduser()
        file_exists = selected_file.is_file()
        file_signature = (
            (selected_file.stat().st_size, selected_file.stat().st_mtime_ns)
            if file_exists
            else (0, 0)
        )
        selection_token = (
            REGION_KEYS[region_name],
            str(selected_file.resolve()),
            *file_signature,
        )
        if st.session_state.get("_home_selection_token") != selection_token:
            eligible_homes = (
                find_eligible_homes(
                    str(selected_file), file_signature[0], file_signature[1]
                )
                if file_exists
                else []
            )
            st.session_state["_eligible_homes"] = eligible_homes
            st.session_state["_home_selection_token"] = selection_token
            if len(eligible_homes) >= 5:
                default_count = min(DEFAULT_HOME_COUNT, len(eligible_homes))
                selected_homes, unseen_home = random_home_selection(
                    eligible_homes, default_count
                )
                st.session_state["home_count"] = default_count
                st.session_state["_generated_home_count"] = default_count
                st.session_state["home_ids_input"] = ", ".join(
                    str(home_id) for home_id in selected_homes
                )
                st.session_state["unseen_test_home"] = unseen_home
            else:
                st.session_state["home_count"] = 0
                st.session_state["_generated_home_count"] = 0
                st.session_state["home_ids_input"] = ""
                st.session_state["unseen_test_home"] = 1

        eligible_homes = st.session_state.get("_eligible_homes", [])
        if len(eligible_homes) >= 5:
            home_count = st.slider(
                "Homes to select (CCP minimum: 5)",
                min_value=5,
                max_value=len(eligible_homes),
                key="home_count",
            )
            if (
                st.session_state.get("_generated_home_count") != home_count
            ):
                selected_homes, unseen_home = random_home_selection(
                    eligible_homes, home_count
                )
                st.session_state["home_ids_input"] = ", ".join(
                    str(home_id) for home_id in selected_homes
                )
                st.session_state["unseen_test_home"] = unseen_home
                st.session_state["_generated_home_count"] = home_count
            if st.button("Randomize home IDs"):
                selected_homes, unseen_home = random_home_selection(
                    eligible_homes, home_count
                )
                st.session_state["home_ids_input"] = ", ".join(
                    str(home_id) for home_id in selected_homes
                )
                st.session_state["unseen_test_home"] = unseen_home
            st.caption(
                f"{len(eligible_homes)} homes in this CSV have complete meter and "
                "target readings with both ON and OFF examples."
            )
        else:
            if file_exists:
                st.warning(
                    f"Only {len(eligible_homes)} eligible home"
                    f"{'' if len(eligible_homes) == 1 else 's'} were found in "
                    f"{selected_file.name}. The experiment needs at least five "
                    "homes with complete grid, air1, furnace1, and drye1 readings "
                    "and both ON/OFF examples. Use a CSV with at least five eligible homes."
                )
            elif region_name in ("New York", "California"):
                eligible_count = {"New York": 4, "California": 0}[region_name]
                st.info(
                    f"The {region_name} CSV is not included in the hosted training "
                    f"data. The supplied dataset has {eligible_count} eligible homes; "
                    "the CCP requires at least five, so this region cannot run the "
                    "current experiment. Select Austin to train."
                )
            else:
                st.info(
                    "No dataset file is available at this path. Select Austin for "
                    "the hosted training cohort, or provide a local CSV path."
                )
        home_text = st.text_input(
            "Home IDs (5 or more)",
            key="home_ids_input",
            disabled=len(eligible_homes) < 5,
        )
        test_home = st.number_input(
            "Unseen test home ID",
            min_value=1,
            key="unseen_test_home",
            disabled=len(eligible_homes) < 5,
        )
        window_size = st.slider("Input history (minutes)", 15, 120, 120, step=15)
        epochs = st.slider("Maximum epochs", 2, 30, 20)
        train_stride = st.select_slider(
            "Training sample stride", options=[1, 2, 5, 10, 20], value=20
        )
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=466)
        with st.expander("Advanced training settings"):
            batch_size = st.select_slider(
                "Batch size", options=[128, 256, 512, 1024], value=512
            )
            patience = st.slider(            "Early-stopping patience", 1, 8, 5)
        st.divider()
        st.markdown(
            '<div class="note">Only the whole-home <b>grid</b> channel is used '
            'as model input. Appliance channels are prediction targets.</div>',
            unsafe_allow_html=True,
        )

    state_col, region_col, model_col = st.columns(3)
    data_status = (
        "Ready"
        if file_exists and len(eligible_homes) >= 5
        else "Too few homes"
        if file_exists
        else "Unavailable"
    )
    with state_col:
        st.markdown(
            f'<div class="mini-card"><div class="label">DATA SOURCE</div>'
            f'<div class="value">{data_status}</div>'
            f'<div class="label">{selected_file.name}</div></div>',
            unsafe_allow_html=True,
        )
    with region_col:
        st.markdown(
            f'<div class="mini-card"><div class="label">REGION</div>'
            f'<div class="value">{region_name}</div>'
            f'<div class="label">Pecan Street · 1-minute data</div></div>',
            unsafe_allow_html=True,
        )
    with model_col:
        st.markdown(
            '<div class="mini-card"><div class="label">MODELS</div>'
            '<div class="value">MLP + CNN + LSTM + GRU</div>'
            '<div class="label">Status and power prediction</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Run an experiment")
    st.write(
        "Train four models with identical splits, then compare their predictions "
        "on chronological and unseen-home test data."
    )
    homes: list[int] = []
    validation_error: str | None = None
    if file_exists and len(eligible_homes) >= 5:
        try:
            homes = parse_home_ids(home_text)
            unavailable_homes = sorted(set(homes) - set(eligible_homes))
            if unavailable_homes:
                validation_error = (
                    "These homes do not have sufficient complete readings for the "
                    f"selected target circuits: {unavailable_homes}"
                )
            if int(test_home) not in homes:
                validation_error = (
                    "The unseen test home must be included in the selected home IDs."
                )
        except ValueError as exc:
            validation_error = str(exc)
        if validation_error:
            st.warning(validation_error)

    run_clicked = st.button(
        "Train all four models",
        type="primary",
        width="stretch",
        disabled=(
            not file_exists or len(eligible_homes) < 5 or bool(validation_error)
        ),
    )
    if run_clicked:
        command = [
            sys.executable,
            "-u",
            str(SCRIPT),
            "--region",
            REGION_KEYS[region_name],
            "--csv",
            str(selected_file),
            "--home-ids",
            *[str(home) for home in homes],
            "--test-home",
            str(int(test_home)),
            "--window",
            str(window_size),
            "--epochs",
            str(epochs),
            "--train-stride",
            str(train_stride),
            "--seed",
            str(int(seed)),
            "--batch-size",
            str(batch_size),
            "--patience",
            str(patience),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
        progress = st.progress(0, text="Starting training process...")
        log_area = st.empty()
        with st.status("Training models. Large CSV files may take several minutes to scan.", expanded=True):
            return_code, output = run_training(command, log_area, progress)
        if return_code == 0:
            progress.progress(1.0, text="Training and evaluation complete.")
            st.success("Experiment completed. Results and charts have been refreshed below.")
            st.rerun()
        else:
            st.error("Training failed. Review the process output below.")
            st.code(output[-10000:], language="text")

    st.divider()
    st.subheader("Latest results")
    metrics, metadata = load_saved_results()
    if metrics is not None:
        render_results(metrics, metadata)
    else:
        st.info("No completed experiment yet. Configure the dataset and run all four models to see results here.")


if __name__ == "__main__":
    main()
