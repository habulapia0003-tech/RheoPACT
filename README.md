# RheoPACT research code and data

This repository accompanies *Digital phenotyping guides process selection for 3D
printing of viscoelastic food inks*. The code connects design-referenced point
cloud phenotyping to visual-quality analysis, rheological characterization and
pre-print process screening.

## Get started

The numerical checks were run with Python 3.13.5. From the repository root,
create a virtual environment and activate it.

```text
python -m venv .venv
```

Use `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate`
in Bash. Then install the dependencies and check the supplied models.

```text
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The verifier compares predictions for the 99-specimen visual-quality test set
and the 121-attempt process-quality test set to the supplied reference values.
It also checks the model's 243-setting process screen. The expected result is
a `PASS` verdict. See [validation and reproduction commands](docs/VALIDATION.md)
for the other numerical analyses.

## Try a process recommendation

```text
python scripts/figure5/predict_process_from_rheology.py --gp 5793 --gpp 664.3 --eta10 110 --eta100 3.18 --tau0 858.34954988555 --K 9.065859201589158e-12 --n 0.01 --top-k 10 --output recommended_processes.csv
```

The example uses seven rheological measurements to score 243 process settings.
The model retains settings with predicted extrusion probability of at least
0.50 and ranks them by predicted objective quality index. The first three
scores are approximately 78.01, 77.36 and 77.12. The full output is in
[examples](examples/expected_top10_processes.csv).

## Repository contents

| Directory | Contents |
| --- | --- |
| `scripts/` | Point-cloud processing, statistical analysis, model training and inference, nozzle-flow analysis and numerical comparisons |
| `models/` | Five fitted model files, including the quality, feasibility and scoring components |
| `data/` | 590 printing attempts, visual-score analysis inputs and partitions, 540 measured rheological conditions, the 131,072-state projection, paired printing records and 11 CAD designs |
| `validation/` | Fixed test inputs and reference tables |
| `docs/` | [Training and inference](docs/TRAINING.md), [figure-to-code index](docs/FIGURE_CODE_MAP.md) and [supporting data](docs/EXTERNAL_DATA.md) |

[Data documentation](data/README.md) describes the tables, material codes and
source records. Workflows using scans, the full reduced-order state table or
COMSOL projects specify their inputs in [supporting data](docs/EXTERNAL_DATA.md).

For enquiries about the code or data, contact Baiyi Lu (bylu@zju.edu.cn).
See [reuse information](NOTICE_NO_LICENSE.md).
