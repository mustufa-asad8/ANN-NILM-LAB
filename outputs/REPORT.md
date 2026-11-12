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
- Input window: 120 consecutive one-minute whole-home readings
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
| train | Air compressor circuit | 61111 | 191453 |
| train | Furnace and air handler | 92877 | 159687 |
| train | Electric clothes dryer | 5238 | 247326 |
| validation | Air compressor circuit | 10627 | 43360 |
| validation | Furnace and air handler | 15193 | 38794 |
| validation | Electric clothes dryer | 1020 | 52967 |
| chronological_test | Air compressor circuit | 13597 | 1032869 |
| chronological_test | Furnace and air handler | 210697 | 835769 |
| chronological_test | Electric clothes dryer | 21194 | 1025272 |
| unseen_test | Air compressor circuit | 59716 | 465765 |
| unseen_test | Furnace and air handler | 67427 | 458054 |
| unseen_test | Electric clothes dryer | 13744 | 511737 |

## Models and training

- Duplicate timestamps are averaged; rows missing a required meter/target value
  are excluded, and sequences are split at missing-minute gaps.
- Appliance targets are clipped at zero to remove negative sensor artifacts.
  Input normalization statistics are fit using training readings only.
- MLP: flattened 120-minute input, two hidden layers, shared encoder, status and
  power output heads.
- 1D-CNN: convolutional encoder over the 120-minute input, with the same two
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
| MLP | Air compressor circuit | 0.730 | 0.906 | 0.915 |
| MLP | Furnace and air handler | 0.655 | 0.834 | 0.821 |
| MLP | Electric clothes dryer | 0.925 | 0.841 | 0.830 |
| 1D-CNN | Air compressor circuit | 0.820 | 0.928 | 0.932 |
| 1D-CNN | Furnace and air handler | 0.760 | 0.875 | 0.872 |
| 1D-CNN | Electric clothes dryer | 0.950 | 0.866 | 0.871 |

## Chronological test results

The final 15% of each of the 14
development homes was kept out of training and validation. Power MAE/RMSE
include appliance-OFF periods.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | Status cutoff | MAE (W) | RMSE (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Air compressor circuit | 0.954 | 0.932 | 0.658 | 0.730 | 121.7 | 491.5 |
| MLP | Furnace and air handler | 0.797 | 0.595 | 0.609 | 0.655 | 118.6 | 238.4 |
| MLP | Electric clothes dryer | 0.988 | 0.829 | 0.841 | 0.925 | 44.1 | 356.3 |
| 1D-CNN | Air compressor circuit | 0.969 | 0.904 | 0.697 | 0.820 | 107.1 | 462.2 |
| 1D-CNN | Furnace and air handler | 0.830 | 0.692 | 0.709 | 0.760 | 109.0 | 217.7 |
| 1D-CNN | Electric clothes dryer | 0.990 | 0.876 | 0.873 | 0.950 | 40.6 | 330.5 |

## Unseen-home results

Home 7951 was excluded completely from training and validation.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | Status cutoff | MAE (W) | RMSE (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Air compressor circuit | 0.909 | 0.887 | 0.814 | 0.730 | 326.9 | 948.7 |
| MLP | Furnace and air handler | 0.897 | 0.863 | 0.805 | 0.655 | 162.3 | 578.5 |
| MLP | Electric clothes dryer | 0.974 | 0.805 | 0.773 | 0.925 | 51.3 | 382.4 |
| 1D-CNN | Air compressor circuit | 0.912 | 0.877 | 0.816 | 0.820 | 288.4 | 832.1 |
| 1D-CNN | Furnace and air handler | 0.894 | 0.831 | 0.790 | 0.760 | 159.6 | 597.7 |
| 1D-CNN | Electric clothes dryer | 0.978 | 0.857 | 0.812 | 0.950 | 83.5 | 603.1 |

### Assignment benchmarks

Classification targets are accuracy >= 0.85, balanced accuracy >= 0.75, and
macro-F1 >= 0.70 (excellent: 0.90, 0.85, and 0.80 respectively). Power error
has no fixed assignment target.

### Minimum benchmark check

#### Chronological test

| Model | Appliance | Accuracy >=85% | Balanced accuracy >=75% | Macro-F1 >=0.70 |
|---|---|---:|---:|---:|
| 1D-CNN | Air compressor circuit | PASS | PASS | FAIL |
| 1D-CNN | Furnace and air handler | FAIL | FAIL | PASS |
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

By average macro-F1 across appliances, 1D-CNN leads the chronological test (0.760), while 1D-CNN leads the unseen-home test (0.806). Individual appliance results vary; compare the tables and error plots for status recognition and power estimation separately.

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
