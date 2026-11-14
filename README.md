# ANN-Based NILM

An ANN-based Non-Intrusive Load Monitoring (NILM) project for the CT-466
Artificial Neural Networks course. It trains and compares four multi-task
architectures (MLP, 1D-CNN, LSTM, and GRU) using only whole-home `grid` power to
predict the ON/OFF state and power of `air1`, `furnace1`, and `drye1`.

Appliance channels are targets only; they are never model inputs. Source
readings are in kW. The committed Austin evaluation holds out home `7951` from
all training and validation. The report evaluates each model and appliance
against the CCP targets; a model is not guaranteed to meet every target on
every held-out split.

## Project layout

```text
data/
  raw/
    austin/       readings_1min.csv, metadata.csv
    california/   readings_1min.csv, metadata.csv
    new_york/     readings_1min.csv, metadata.csv
    puerto_rico/  monthly measurement files and metadata
  hosted/
    austin_eligible_homes.csv.gz
src/
  nilm_ccp/
    dashboard.py
    train.py
app.py                 Streamlit Cloud entrypoint
tests/
  test_training.py
artifacts/
  checkpoints/
  figures/
  logs/
  metadata/
  metrics/
  reports/
```

Full regional source data is kept out of the application package and is not
copied into the Docker image. The compact Austin cohort is included for cloud
training. See [data/README.md](data/README.md) for the supplied file formats.

## Requirements and setup

Python 3.10 or newer is recommended. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` installs the dashboard, data-analysis, and CPU-only PyTorch
dependencies so Streamlit Cloud can train without downloading unused CUDA
packages. `requirements-training.txt` is a convenience include for local
training.

Run the focused regression tests with:

```powershell
python -m unittest discover -s tests -v
```

## Run the dashboard

```powershell
streamlit run src\nilm_ccp\dashboard.py
```

The deployed Streamlit Community Cloud app uses the root `app.py` entrypoint.
It trains on the hosted Austin cohort and also displays the committed Austin
evaluation artifacts. The app is available at
[ann-nilm-lab-ct466.streamlit.app](https://ann-nilm-lab-ct466.streamlit.app/).
To connect a new deployment, select this GitHub repository, branch `main`, and
main file path `app.py`.

Choose a supported region, eligible homes, a held-out home, and training
settings in the sidebar. The dashboard scans the selected CSV for complete
`grid`/target records and homes with ON and OFF examples. It randomly selects
five eligible homes by default; the unseen-home test uses one complete
selected home that is excluded from fitting and calibration.

Only Austin currently has enough eligible households to meet the CCP’s
five-home minimum (15 eligible homes in the supplied dataset). New York has
four qualifying homes and California has none for the selected circuits, so
they cannot run this experiment even if their full CSVs are available. The
supplied Puerto Rico data is organized as monthly measurements rather than the
wide household CSV format and is not currently supported.

## Train and evaluate

The default CLI run uses the committed 15-home Austin cohort, a 120-minute
input window, 20 training epochs maximum, batch size 512, early-stopping
patience 5, training stride 20, and seed 466:

```powershell
python src\nilm_ccp\train.py --region austin
```

Regenerate metrics, figures, and the report without retraining:

```powershell
python src\nilm_ccp\train.py --region austin --evaluate-only
```

Evaluation-only mode requires the saved cohort and training settings to match,
and checks that the checkpoint normalization and ON thresholds match the
current training data. To use a different region, CSV, cohort, or training
configuration, omit `--evaluate-only` and train new checkpoints. For example:

```powershell
python src\nilm_ccp\train.py --region austin --epochs 15
python src\nilm_ccp\train.py --csv "C:\data\readings.csv" --home-ids 558 3488 950 1417 5587 --test-home 5587
```

Each selected home must provide a whole-home `grid` channel and all three
appliance channels. The pipeline averages duplicate timestamps, removes
incomplete readings, splits at missing-minute gaps, and creates windows only
within a split. The chronological split is 70% training, 15% validation, and
15% test for each development home. At least one other complete home is
reserved for unseen-home testing. Thresholds, input normalization, class
weights, and status cutoffs are fit using training or validation data only;
the test labels are not used for calibration.

## CCP evaluation

The required held-out classification benchmarks are accuracy >= 0.85,
balanced accuracy >= 0.75, and macro-F1 >= 0.70 for each of three appliances.
The excellent targets are 0.90, 0.85, and 0.80. Power MAE/RMSE are reported
but have no fixed benchmark. No architecture can guarantee passing on every
held-out home; scores must be measured on data not used for training or
threshold selection. The report includes per-appliance class counts,
confusion matrices, actual-versus-predicted power plots, model comparison,
data-cleaning details, assumptions, and limitations.

## Artifacts

Training and the dashboard update the following versioned deliverables under
`artifacts/`:

- `metrics/metrics.csv`: chronological and unseen-home metrics.
- `reports/report.md`: methodology, comparison, and benchmark checks.
- `figures/`: confusion matrices and power prediction plots.
- `logs/training_log.csv`: training and validation metrics by epoch.
- `metadata/run_metadata.json`: cohort, thresholds, split sizes, versions, and
  seed.
- `checkpoints/`: validation-selected MLP, 1D-CNN, LSTM, and GRU checkpoints
  after a four-model training run.
- `nilm_ccp_submission.zip`: submission bundle available from the
  dashboard.

The dashboard also offers downloads for the report, metrics, and complete
submission bundle.

## Docker Compose

From the project directory:

```powershell
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501). The source data is mounted
read-only from `data/raw`; generated artifacts persist in the local
`artifacts/` directory. Inspect logs and stop the app with:

```powershell
docker compose logs -f
docker compose down
```

## Notes

The supplied `grid` channel is whole-home net-meter power, not gross household
consumption; it can be negative when solar generation exceeds demand. Metrics
and conclusions should be interpreted with this limitation in mind.
