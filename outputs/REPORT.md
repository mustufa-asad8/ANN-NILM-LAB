# NILM ANN CCP Results

## Project

Design and implementation of an ANN-based Non-Intrusive Load Monitoring system.
The models use only the whole-home `grid` meter history to estimate three
appliance ON/OFF states and appliance power. Appliance channels are targets,
never model inputs. Source readings are kW and reported power errors are W.

## Dataset and split

- Region file: austin
- Selected homes: 7951, 8565, 2361, 6139, 7536, 8156, 9019, 4767, 3456, 7901, 4373, 2335, 9922, 9278, 661
- Training/validation homes: 8565, 2361, 6139, 7536, 8156, 9019, 4767, 3456, 7901, 4373, 2335, 9922, 9278, 661
- Unseen-home test: 7951 (complete held-out home; no training or threshold fitting)
- Input column: `grid`; target columns: `air1`, `furnace1`, `drye1`
- Input window: 60 consecutive one-minute whole-home readings
- Per development home: chronological 70% train, 15% validation, and 15%
  chronological test. The complete final home is a separate unseen-home test.
- Windows do not cross split or missing-data boundaries.
- Seed: 466

## Appliance ON thresholds

Thresholds are computed using positive readings from training portions only,
then applied unchanged to validation and test.

- Air compressor circuit: 50.0 W (max(50 W, 10th percentile of training readings >20 W))
- Furnace and air handler: 50.0 W (max(50 W, 10th percentile of training readings >20 W))
- Electric clothes dryer: 357.0 W (max(50 W, 10th percentile of training readings >20 W))

## Thresholded ON/OFF example counts

Training and validation counts use the configured training stride; test counts
include every scored minute.

| Split | Appliance | ON examples | OFF examples |
|---|---|---:|---:|
| train | Air compressor circuit | 61130 | 191556 |
| train | Furnace and air handler | 92934 | 159752 |
| train | Electric clothes dryer | 5241 | 247445 |
| validation | Air compressor circuit | 10667 | 43415 |
| validation | Furnace and air handler | 15238 | 38844 |
| validation | Electric clothes dryer | 1020 | 53062 |
| chronological_test | Air compressor circuit | 13690 | 1043390 |
| chronological_test | Furnace and air handler | 212258 | 844822 |
| chronological_test | Electric clothes dryer | 21398 | 1035682 |
| unseen_test | Air compressor circuit | 59716 | 465825 |
| unseen_test | Furnace and air handler | 67438 | 458103 |
| unseen_test | Electric clothes dryer | 13744 | 511797 |

## Models and training

- Duplicate timestamps are averaged; rows missing a required meter/target value
  are excluded, and sequences are split at missing-minute gaps.
- Appliance targets are clipped at zero to remove negative sensor artifacts.
  Input normalization statistics are fit using training readings only.
- MLP: flattened 60-minute input, two hidden layers, shared encoder, status and
  power output heads.
- 1D-CNN: convolutional encoder over the 60-minute input, with the same two
  output tasks.
- Both models use class-weighted binary cross entropy plus smooth L1 power loss.
  The best checkpoint is selected by mean validation macro-F1, with mean
  balanced accuracy as a tie-breaker.
- Each model's ON/OFF probability cutoff is selected separately per appliance
  from 0.05 to 0.95 in 0.005 steps by maximizing validation macro-F1, with
  balanced accuracy as a tie-breaker.
  The cutoffs are then frozen before either test set is evaluated.
- Training samples use a stride of 20 minutes to reduce
  redundancy; both test sets are evaluated every minute.
- Maximum epochs: 20; early-stopping patience: 5; batch size: 512.

### Validation-only status cutoff calibration

| Model | Appliance | Status cutoff | Validation macro-F1 | Validation balanced accuracy |
|---|---|---:|---:|---:|
| MLP | Air compressor circuit | 0.775 | 0.904 | 0.913 |
| MLP | Furnace and air handler | 0.660 | 0.833 | 0.825 |
| MLP | Electric clothes dryer | 0.945 | 0.847 | 0.842 |
| 1D-CNN | Air compressor circuit | 0.740 | 0.914 | 0.920 |
| 1D-CNN | Furnace and air handler | 0.605 | 0.859 | 0.861 |
| 1D-CNN | Electric clothes dryer | 0.950 | 0.870 | 0.876 |

## Chronological test results

The final 15% of each of the 14
development homes was kept out of training and validation. Power MAE/RMSE
include appliance-OFF periods.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | Status cutoff | MAE (W) | RMSE (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Air compressor circuit | 0.950 | 0.926 | 0.647 | 0.775 | 133.1 | 505.9 |
| MLP | Furnace and air handler | 0.796 | 0.622 | 0.636 | 0.660 | 119.5 | 241.4 |
| MLP | Electric clothes dryer | 0.989 | 0.834 | 0.849 | 0.945 | 43.8 | 349.8 |
| 1D-CNN | Air compressor circuit | 0.958 | 0.938 | 0.668 | 0.740 | 118.4 | 486.1 |
| 1D-CNN | Furnace and air handler | 0.810 | 0.648 | 0.664 | 0.605 | 114.4 | 234.0 |
| 1D-CNN | Electric clothes dryer | 0.990 | 0.874 | 0.870 | 0.950 | 41.8 | 340.5 |

## Unseen-home results

Home 7951 was excluded completely from training and validation.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | Status cutoff | MAE (W) | RMSE (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Air compressor circuit | 0.909 | 0.882 | 0.814 | 0.775 | 294.5 | 824.2 |
| MLP | Furnace and air handler | 0.890 | 0.863 | 0.797 | 0.660 | 166.1 | 606.6 |
| MLP | Electric clothes dryer | 0.970 | 0.836 | 0.768 | 0.945 | 66.5 | 438.5 |
| 1D-CNN | Air compressor circuit | 0.916 | 0.896 | 0.826 | 0.740 | 306.5 | 895.4 |
| 1D-CNN | Furnace and air handler | 0.889 | 0.875 | 0.799 | 0.605 | 159.5 | 614.3 |
| 1D-CNN | Electric clothes dryer | 0.982 | 0.892 | 0.843 | 0.950 | 58.5 | 413.3 |

### Assignment benchmarks

Classification targets are accuracy >= 0.85, balanced accuracy >= 0.75, and
macro-F1 >= 0.70 (excellent: 0.90, 0.85, and 0.80 respectively). Power error
has no fixed assignment target.

### Minimum benchmark check

#### Chronological test

| Model | Appliance | Accuracy >=85% | Balanced accuracy >=75% | Macro-F1 >=0.70 |
|---|---|---:|---:|---:|
| 1D-CNN | Air compressor circuit | PASS | PASS | FAIL |
| 1D-CNN | Furnace and air handler | FAIL | FAIL | FAIL |
| 1D-CNN | Electric clothes dryer | PASS | PASS | PASS |
| MLP | Air compressor circuit | PASS | PASS | FAIL |
| MLP | Furnace and air handler | FAIL | FAIL | FAIL |
| MLP | Electric clothes dryer | PASS | PASS | PASS |

- 1D-CNN: below at least one minimum for Air compressor circuit, Furnace and air handler.
- MLP: below at least one minimum for Air compressor circuit, Furnace and air handler.

#### Unseen-home test

| Model | Appliance | Accuracy >=85% | Balanced accuracy >=75% | Macro-F1 >=0.70 |
|---|---|---:|---:|---:|
| 1D-CNN | Air compressor circuit | PASS | PASS | PASS |
| 1D-CNN | Furnace and air handler | PASS | PASS | PASS |
| 1D-CNN | Electric clothes dryer | PASS | PASS | PASS |
| MLP | Air compressor circuit | PASS | PASS | PASS |
| MLP | Furnace and air handler | PASS | PASS | PASS |
| MLP | Electric clothes dryer | PASS | PASS | PASS |

- 1D-CNN: meets the minimum for all three appliances.
- MLP: meets the minimum for all three appliances.

## Comparison and conclusion

By average macro-F1 across appliances, 1D-CNN leads the chronological test (0.734), while 1D-CNN leads the unseen-home test (0.823). Individual appliance results vary; compare the tables and error plots for status recognition and power estimation separately.

Models meeting all appliance minimums: chronological test: none; unseen-home test: 1D-CNN, MLP.

## Figures and training log

- `outputs/chronological_test_confusion_matrices.png`
- `outputs/chronological_test_power_predictions.png`
- `outputs/unseen_home_confusion_matrices.png`
- `outputs/unseen_home_power_predictions.png`
- `outputs/training_log.csv`

## Reproducibility and limitations

Run `python nilm_ccp.py --region austin`. The selected cohort and settings can
be overridden from the command line. The chronological test measures later-time
performance on the development homes; the unseen-home test measures transfer to
one unseen home, not all Pecan Street homes. `grid` is the available whole-home
net-meter reading; it can be negative when a home exports solar power, so the
input is net grid power rather than gross appliance-sum consumption. This
distinction should be included when discussing limitations.
