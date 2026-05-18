# Entropy-Driven HE Financial Planning — Streamlit Demo

Interactive prototype for the framework described in:

> Schetinin, V. (2026). *Enrollment Management under Deep Uncertainty:
> An Entropy-Indexed POMDP Framework for UK Higher Education Finance.*
> Socio-Economic Planning Sciences (under review).
> SSRN preprint: https://doi.org/10.2139/ssrn.6308238

---

## What the app does

| Tab | Description |
|-----|-------------|
| 📅 Year-by-Year | Step through a 12-year planning horizon manually. Record your institution's actual enrolment signal each year and watch the Bayesian belief update in real time. |
| 📈 12-Year Simulation | Run all 6 strategies on a shared observation path (common random numbers — fair comparison). Inspect cumulative reward curves, action maps, and entropy trajectory. |
| ⚖️ Strategy Comparison | Monte Carlo comparison of all 6 strategies. Reproduces Table 4 (n = 500, seed = 42): Unified Framework = 349.2 reward · 1.1 % crisis probability. Pareto dominance scatter included. |
| 📂 Calibrate Your Data | Upload a CSV (`year`, `enrollment_total`) to re-estimate T and Ω via the same Laplace-smoothed pipeline as the paper. Activate to replace Russell Group parameters across all tabs. |

Sidebar sliders control all framework hyperparameters (γ, δ, U_min, λ_VoI, μ_OV, n_sat, b₀). All defaults match the paper (§5, Table 3).

---

## Mac Quick Start (recommended for Mac users)

`Launch FinPlanner.command` in the project root is a double-clickable launcher
that handles the Python environment automatically.

### One-off setup (in Terminal)

```bash
brew install uv
```

[Homebrew](https://brew.sh) must be installed first if not already present
(`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`).

### First launch

1. Download or clone the repository.
2. If macOS blocks the file: **right-click → Open** (Gatekeeper warning; once only).
3. Double-click `Launch FinPlanner.command`.
4. A Terminal window opens; on first run it creates the Python environment (~30 s).
5. The browser opens automatically at `http://localhost:8501`.

### Subsequent launches

Double-click `Launch FinPlanner.command` — the app starts in ~3 seconds.

### To stop the app

Close the Terminal window (or press `Ctrl-C` inside it).

### Data security

All processing runs on your Mac (`localhost:8501`).  No data leaves your machine.
Disconnect Wi-Fi — the app still works.

---

## Running locally (any platform)

```bash
# From the project root
pip install -r DEMO/requirements.txt
streamlit run DEMO/app.py
```

First load: ~2 s (loads `paper_params.npz` cache).  
Subsequent interactions: < 1 s.

---

## Deploying to Streamlit Community Cloud

### Prerequisites

1. Push the repository to a **public** GitHub account.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.

### Required repo layout

```
/ (repo root)
├── .streamlit/
│   └── config.toml          ← theme and server settings
├── DEMO/
│   ├── app.py               ← Streamlit entry point
│   ├── pomdp_core.py        ← POMDP class and calibration wrapper
│   ├── requirements.txt     ← pinned dependencies
│   └── paper_params.npz     ← 2 KB pre-computed parameter cache (MUST be committed)
└── rg_pomdp_calibration.py  ← calibration pipeline (imported by pomdp_core.py)
```

### Deployment steps

1. In the Community Cloud dashboard click **New app**.
2. Select your repository and branch.
3. Set **Main file path** to `DEMO/app.py`.
4. Click **Deploy**.

Community Cloud will install from `DEMO/requirements.txt` and start the app.
On first load it reads `DEMO/paper_params.npz` (fast, ~0.1 s) instead of running
the full HESA calibration pipeline, so the app always reproduces the paper's
calibrated parameters regardless of whether data files are present.

### If HESA data files are present

The calibration pipeline in `rg_pomdp_calibration.py` reads:

| File | Description |
|------|-------------|
| `data_1/table-1.zip` | HESA enrolment CSVs (2014/15–2024/25) |
| HESA finance CSV | Table 5 net operating margin by institution |
| ONS National Population Projections | Demographic prior |
| Home Office visa data | International student adjustment |

If these files are absent the pipeline falls back to synthetic data automatically.
The pre-computed cache (`paper_params.npz`) already contains the real parameters,
so the fallback is never triggered in a deployed app that commits the cache.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.57.0 | Web framework |
| numpy | 2.4.3 | Numerics |
| scipy | 1.17.1 | SR3 MC rollouts |
| matplotlib | 3.10.8 | All figures |
| pandas | 2.3.3 | DataFrames and CSV parsing |
| hmmlearn | 0.3.3 | HMM fallback in calibration |
| openpyxl | 3.1.5 | HESA Excel loading |

---

## Framework parameters (paper defaults)

| Parameter | Value | Role |
|-----------|-------|------|
| γ | 0.95 | Discount factor |
| δ | 0.10 | SR3 risk tolerance |
| U_min | −25 | Satisficing floor |
| λ_VoI | 1.5 | SR4 VoI bonus weight |
| μ_OV | 1.2 | SR5 option-value penalty weight |
| n_sat | 30 | SR3 MC rollouts |
| b₀ | [0.159, 0.730, 0.111] | Pooled initial belief (Low/Stable/Growth) |

---

## Citation

```bibtex
@article{schetinin2026entropy,
  author  = {Schetinin, Vitaly},
  title   = {Enrollment Management under Deep Uncertainty:
             An Entropy-Indexed POMDP Framework for UK Higher Education Finance},
  journal = {Socio-Economic Planning Sciences},
  year    = {2026},
  note    = {Under review. SSRN preprint doi:10.2139/ssrn.6308238}
}
```
