# Dataset files

Keep the instructor-provided source files in `data/raw/` using these paths:

| Region | Data file | Metadata |
|---|---|---|
| Austin | `austin/readings_1min.csv` | `austin/metadata.csv` |
| California | `california/readings_1min.csv` | `california/metadata.csv` |
| New York | `new_york/readings_1min.csv` | `new_york/metadata.csv` |
| Puerto Rico | Monthly metric CSVs (`*.csv.gz`) | `puerto_rico/metadata.csv` |

The Austin, California, and New York wide CSV files contain one-minute
whole-home and appliance readings and can be selected in the dashboard or
training script. Puerto Rico is supplied as separate monthly measurements
(angle, apparent power, current, real power, and total harmonic distortion);
that layout is retained as received but is not yet a supported input format
for the NILM training pipeline.

The raw files are large and excluded from Docker build context and source
control. Do not rename or edit their column headers; the experiment expects
`dataid`, `localminute`, `grid`, `air1`, `furnace1`, and `drye1` in the
supported wide CSV files.
