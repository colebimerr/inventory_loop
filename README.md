# Inventory Variance Loop Accelerator

Shorten the detection-to-resolution loop on inventory variance for multi-branch HVAC distributors.

This is the v1 demo app: a single-user Streamlit prototype that runs on three CSVs. Prospects can click "Use demo data" or upload their own three files. Three views: Prioritized Count List, Variance Queue, ROI Calculator.

---

## Quickstart (local)

```bash
# 1. From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the demo dataset (writes to ./data)
python generate_data.py

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501.

The app starts on the upload screen. Click **Load demo data** to walk through with the synthetic dataset, or upload your own three CSVs (see schema below).

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file to `app.py`. Streamlit Cloud will install `requirements.txt` automatically.
4. After deploy, the demo dataset is already baked into `/data` so prospects can click "Load demo data" without uploading anything.

---

## Data schema

The app accepts three CSVs. Schemas are validated on upload.

### `inventory_master.csv`
| Column | Type | Notes |
|---|---|---|
| sku_id | string (PK) | |
| description | string | |
| category | string | One of: Refrigerant, Copper Tubing, Compressor, Motor, Controls, Miscellaneous |
| unit_cost | float | USD |
| velocity_class | string | A / B / C |
| reorder_point | int | |
| reorder_qty | int | |
| last_cycle_count_date | ISO date | |

### `variance_events.csv`
| Column | Type | Notes |
|---|---|---|
| event_id | string (PK) | |
| sku_id | FK | → inventory_master |
| branch_id | FK | → branches |
| detection_source | string | cycle_count / picking / stocking / branch_escape |
| detection_date | ISO datetime | |
| resolution_date | ISO datetime | Null if still open |
| expected_qty | int | |
| actual_qty | int | |
| variance_qty | int | signed |
| variance_cost | float | signed |
| stage_history | JSON string | array of {stage, started_at, completed_at} |
| assigned_to | string | |
| current_status | string | open / investigating / physical_count_complete / system_corrected / resolved |

### `branches.csv`
| Column | Type | Notes |
|---|---|---|
| branch_id | string (PK) | |
| branch_name | string | |
| branch_type | string | DC / branch |
| city | string | |
| state | string | |

---

## File structure

```
.
├── app.py                      Streamlit entry + landing page
├── generate_data.py            Synthetic data generator (run to refresh /data)
├── requirements.txt
├── data/                       Generated CSVs (gitignored if you prefer)
├── views/
│   ├── variance_queue.py       View 2 (build-anchor view)
│   ├── count_list.py           View 1
│   └── roi.py                  View 3
├── logic/
│   ├── loop_metrics.py         Loop-time + stage-timing math
│   ├── risk_scoring.py         Per-SKU weighted risk heuristic (<30 lines of core logic)
│   └── roi_calc.py             ROI math + data-driven defaults
└── utils/
    └── csv_validators.py       Schema + foreign-key checks
```

---

## Regenerating the demo dataset

Tweak the dataclass at the top of `generate_data.py`:

```python
@dataclass
class Config:
    n_skus: int = 2500
    history_days: int = 180
    variance_events_total: int = 1800
    open_event_ratio: float = 0.10
    problem_branch_multiplier: float = 2.0
    ...
```

Then `python generate_data.py` writes fresh CSVs in place.

---

## What's NOT in v1

By design:

- No authentication, no user accounts
- No persistence between sessions
- No database — pure CSV
- No real ML for risk scoring — readable weighted heuristic only
- No ERP integration (Eclipse/Epicor integration is phase 2)
- No alerting / digest emails (phase 2)

The architecture (separated data / logic / views) is set up so those can be added incrementally without rewrites.
