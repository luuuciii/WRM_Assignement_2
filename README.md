# WRM Module 2 — Timeseries Analysis

Hydrological timeseries analysis for the Ill River (Gisingen, Austria) and Rhine River (Diepoldsau, Switzerland). Covers discharge Q and suspended sediment concentration C from sub-hourly records spanning 1976–2025.

## Structure

```
├── Code/
│   ├── main.ipynb      # Main notebook — all analysis, plots, and discussion
│   └── helpers.py      # All helper functions (imported by the notebook)
├── DATA/
│   ├── Q_Gisingen_1976-2023.csv
│   ├── SSC_Gisingen_2003-2020.csv
│   ├── Q_Diepoldsau_m3s.csv
│   └── SSC_Diepoldsau_gL.csv
└── Module2_Lab.pdf     # Assignment brief
```

## Tasks

| Section | Task | Points |
|---------|------|--------|
| 1 | Timeseries Review — load, resample, trend test, detrend | 2 |
| 2 | Timeseries Modelling — seasonal adjustment, ACF/PACF, AR/ARMA fitting | 5 |
| 3 | Application & Evaluation — theoretical ACF, residual diagnostics, normality | 5 |
| 4 | Simulation — synthetic series, sediment mass flux, Ill/Rhine ratio | 5 |
| 5 | Independence Test — Q–C correlation, chi-squared test | 3 |

## Methodology

**Pre-processing pipeline** (applied before all modelling):
1. Resample sub-hourly data to monthly means
2. Log-transform: `y(t) = log(x(t))`
3. Remove linear trend in log-space (OLS, α = 5%)
4. Monthly standardisation: subtract log-space climatological mean, divide by log-space std (Q series only)

**Models fitted** (chosen by AIC/BIC grid search):

| Series | AR order | ARMA order |
|--------|----------|------------|
| Gisingen Q   | AR(1)  | ARMA(1,1) |
| Gisingen C   | AR(1)  | ARMA(2,2) |
| Diepoldsau Q | AR(3)  | ARMA(1,1) |
| Diepoldsau C | AR(1)  | ARMA(1,1) |

**Key results:**
- Log transform resolves both residual autocorrelation (Ljung-Box) and non-normality (PPCC) — all models pass both tests after transformation
- Q and C are strongly positively correlated (Pearson r ≈ 0.47–0.58, Spearman ρ ≈ 0.68–0.74); independent simulation underestimates peak sediment events

## Setup

```bash
conda activate wrm          # or your environment name
cd Code
jupyter notebook main.ipynb
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`

## Git Workflow

One branch per task (`task1-review`, `task2-modelling`, etc.), each merged to `main` via pull request. All logic lives in `helpers.py`; `main.ipynb` only calls functions and displays results.
