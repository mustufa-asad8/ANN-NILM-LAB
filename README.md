# ANN-Based NILM CCP

This project trains and compares a multi-task MLP and 1D-CNN for Non-Intrusive
Load Monitoring (NILM). Both models use only the household whole-home `grid`
meter history to estimate `air1`, `furnace1`, and `drye1`
ON/OFF states and power. The appliance channels are targets only; they are
never model inputs.

## Dataset

The supplied workspace includes New York, California, and Austin 1-minute CSV
files and Pecan Street metadata. The default experiment uses Austin because it
contains a larger cohort with complete readings for the selected circuits:

`1minute_data_austin/1minute_data_austin.csv`

The default cohort contains 15 homes: `7951`, `8565`, `2361`, `6139`, `7536`,
`8156`, `9019`, `4767`, `3456`, `7901`, `4373`, `2335`, `9922`, `9278`, and
`661`. Home `7951` is reserved for unseen-home testing; the other 14 homes
provide a more diverse training and validation cohort. The target circuit
channels are `air1` (air compressor circuit),
`furnace1` (furnace and air handler), and `drye1` (electric clothes dryer);
all selected homes have aligned readings and ON/OFF examples for each channel.
Each of the development homes is split chronologically into 70% training, 15%
validation, and 15% test. Adjacent windows never cross split or missing-data
boundaries.

All source power readings are in kW. For each appliance, the ON threshold is
computed from positive training-part readings as the larger of 50 W and their
10th percentile. Thresholds, input normalization, and positive-class weights
are fit from training data only.

## Run

Install the requirements if needed:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
```

Launch the graphical dashboard:

```powershell
streamlit run app.py
```

In the sidebar, choose the dataset, homes, held-out home, and training settings.
Click **Train MLP and 1D-CNN** to run both models; the dashboard displays live
training output, metrics with target PASS/FAIL checks, confusion matrices, power
plots, and a downloadable report or complete submission bundle.

When a region or CSV path changes, the dashboard scans that file for homes with
at least 10,000 complete `grid`/target readings and at least 100 ON and 100 OFF
examples for each target circuit (using 50 W as the eligibility boundary). It
then randomly selects up to 15 eligible homes and one of those homes for the
unseen-home test. Use **Randomize home IDs** to draw another cohort, or adjust
the cohort-size slider to select a different number. Home IDs can also be edited
manually, but training is disabled if the selected cohort contains an ineligible
home or the unseen-test home is not in that cohort. Regions with fewer than five
eligible homes are reported as unavailable for this experiment rather than
filled with unsuitable homes.

With the supplied files and these criteria, the current scan finds 15 eligible
Austin homes, four in New York, and none in California. Austin is therefore the
only supplied region that currently meets the five-home minimum for this
three-circuit experiment; the counts are recalculated when the source CSV
changes.

## Public dashboard

The public Streamlit deployment displays the committed Austin evaluation
results, report, and charts. The multi-gigabyte regional CSVs are intentionally
not hosted with the app, so starting new training runs is available only in a
local or Docker setup where the dataset files are present. The base
`requirements.txt` contains the dashboard dependencies; PyTorch is kept in
`requirements-training.txt` so the public read-only view does not download the
large training framework.

## Docker Compose

Build the image and start the dashboard from the project directory:

```powershell
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501). Check startup and training
logs with:

```powershell
docker compose logs -f
```

Stop and remove the container and Compose network with:

```powershell
docker compose down
```

The regional dataset folders are mounted read-only into the container. Generated
reports, metrics, charts, and model checkpoints are stored in the local
`outputs/` folder and remain available after the container is stopped or
removed. The image build deliberately excludes the large CSV datasets; keep the
dataset folders beside `compose.yaml`.

Train both networks and create the report/figures using the expanded Austin
cohort:

```powershell
python nilm_ccp.py --region austin
```

After training, regenerate metrics and figures from the saved model checkpoints
without retraining:

```powershell
python nilm_ccp.py --region austin --evaluate-only
```

Use another region, dataset path, home cohort, or training duration with
arguments, for example:

```powershell
python nilm_ccp.py --region austin --epochs 15
python nilm_ccp.py --csv "path\to\your.csv" --home-ids 558 3488 950 1417 5587 --test-home 5587
```

Every selected home must have the `grid` channel and all three appliance
channels. The script validates this through actual non-missing, contiguous
minute readings and fails with an explicit error if a required home or usable
split is absent.

## Outputs

The `outputs/` directory contains:

- `metrics.csv`: accuracy, balanced accuracy, macro-F1, MAE, and RMSE for both
  chronological and unseen-home test sets.
- `REPORT.md`: methodology and the measured comparison table.
- `chronological_test_confusion_matrices.png` and `unseen_home_confusion_matrices.png`:
  per-model, per-appliance ON/OFF confusion matrices.
- `chronological_test_power_predictions.png` and `unseen_home_power_predictions.png`:
  actual and estimated power for the first 24 test hours.
- `training_log.csv`: train and validation losses for every epoch.
- `run_metadata.json`: cohort, thresholds, split sizes, versions, and seed.
- `mlp_best.pt` and `1d_cnn_best.pt`: checkpoints selected by validation
  macro-F1, with balanced accuracy as a tie-breaker.

For status predictions, each model/appliance probability cutoff is selected
from the pooled validation split by searching 0.05 to 0.95 in increments of
0.005, maximizing macro-F1, with balanced accuracy as a tie-breaker. The
selected cutoff is frozen before either test set is evaluated; the held-out
home's labels are not used for calibration. Benchmarks
from the CCP are accuracy >=85%, balanced accuracy >=75%, and macro-F1 >=0.70;
excellent targets are >=90%, >=85%, and >=0.80. Power MAE and RMSE should also
be reported but have no fixed benchmark.

## Notes

The `grid` channel is the supplied whole-home net-meter reading. Because it may
be negative while a home exports solar power, it is not identical to gross
household consumption. This is documented in the generated report. The chosen
home IDs have full appliance-channel coverage and enough ON/OFF examples in the
provided Austin CSV; the script calculates and reports the final thresholded
test-class counts for each run.
