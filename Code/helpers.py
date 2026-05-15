# =============================================================================
# helpers.py — Helper functions for WRM Module 2 Timeseries Analysis
# =============================================================================
# Each section imports only the functions it needs from this file.
# Every function documents: what it does, inputs (type + meaning),
# and outputs (type + meaning).
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# SECTION 1 — Timeseries Review
# =============================================================================

def load_monthly(filepath, value_col):
    """
    Load a sub-hourly timeseries CSV and resample to monthly means.

    Reads a CSV with a 'timestamp' column and resamples to calendar-month
    averages. Months with no valid readings are returned as NaN.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    value_col : str
        Name of the column containing the numeric values (e.g. 'q_m3s').

    Returns
    -------
    pd.Series
        Monthly mean values with a DatetimeIndex (month-start frequency).
        Index name: 'timestamp', Series name: value_col.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    monthly = df[value_col].resample("MS").mean()
    return monthly


def fit_linear_trend(series):
    """
    Fit an OLS linear regression to a timeseries and test slope significance.

    Uses integer time index (0, 1, 2, ...) so the slope is in units/month.
    Performs a two-sided t-test on the slope coefficient.

    Parameters
    ----------
    series : pd.Series
        Monthly timeseries with DatetimeIndex. NaN values are dropped before fitting.

    Returns
    -------
    dict with keys:
        slope       (float) : Slope in [units / month].
        intercept   (float) : Intercept at t=0 (first valid observation).
        p_value     (float) : Two-sided p-value for the slope (H0: slope = 0).
        t_stat      (float) : t-statistic of the slope.
        r_squared   (float) : Coefficient of determination R².
        std_err     (float) : Standard error of the slope estimate.
        fitted      (pd.Series) : Fitted trend values aligned to the valid index.
    """
    valid = series.dropna()
    x = np.arange(len(valid), dtype=float)
    y = valid.values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    t_stat = slope / std_err

    fitted = pd.Series(intercept + slope * x, index=valid.index, name=series.name)

    return {
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "t_stat": t_stat,
        "r_squared": r_value ** 2,
        "std_err": std_err,
        "fitted": fitted,
    }


def detrend_series(series, trend_result, alpha=0.05):
    """
    Remove the mean (or a statistically significant linear trend) from a
    timeseries, returning a zero-mean normalised series z(t).

    If the slope p-value < alpha the fitted linear trend is subtracted first,
    then the residual mean is removed to guarantee zero mean.
    If the slope is not significant, only the sample mean is subtracted.

    Parameters
    ----------
    series : pd.Series
        Monthly timeseries with DatetimeIndex. NaN values are dropped.
    trend_result : dict
        Output of fit_linear_trend().
    alpha : float
        Significance level for the slope test (default 0.05).

    Returns
    -------
    z : pd.Series
        Zero-mean normalised timeseries (same index as valid observations).
    sigma2 : float
        Sample variance of z (σ²_z).
    trend_removed : bool
        True if a significant linear trend was subtracted.
    """
    valid = series.dropna()
    trend_removed = trend_result["p_value"] < alpha

    if trend_removed:
        residual = valid - trend_result["fitted"]
    else:
        residual = valid.copy()

    z_valid = residual - residual.mean()
    sigma2  = float(z_valid.var(ddof=1))

    # Reindex back to the original series index so NaN gaps are preserved.
    # This is important for segment-based ACF/PACF computation in Section 2.
    z = z_valid.reindex(series.index)

    return z, sigma2, trend_removed


def summarise_trend(label, trend_result, alpha=0.05):
    """
    Build a one-row summary dict for printing trend test results.

    Parameters
    ----------
    label : str
        Human-readable name for the series (e.g. 'Gisingen Q').
    trend_result : dict
        Output of fit_linear_trend().
    alpha : float
        Significance level used for the decision (default 0.05).

    Returns
    -------
    dict
        Keys: label, slope, p_value, r_squared, significant.
    """
    sig = trend_result["p_value"] < alpha
    return {
        "Series": label,
        "Slope (units/month)": f"{trend_result['slope']:.4e}",
        "p-value": f"{trend_result['p_value']:.4f}",
        "R²": f"{trend_result['r_squared']:.4f}",
        "Significant (5%)": "YES" if sig else "no",
    }


# =============================================================================
# SECTION 2 — Timeseries Modelling
# =============================================================================

from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.arima.model import ARIMA


def remove_seasonal_means(series, standardise=True):
    """
    Remove the seasonal cycle from a monthly series by subtracting the
    climatological mean for each calendar month, and optionally dividing by
    the climatological standard deviation (monthly standardisation).

    Parameters
    ----------
    series : pd.Series
        Monthly timeseries with DatetimeIndex. May contain NaN.
    standardise : bool
        If True (default), also divide by the monthly std, producing a
        dimensionless anomaly [–] with unit variance per calendar month.
        Set to False for series with very small monthly std values (e.g. C).

    Returns
    -------
    anomaly : pd.Series
        Seasonally adjusted series. Dimensionless [–] if standardise=True,
        original units if standardise=False.
    clim : pd.Series
        Monthly climatological mean, index = 1..12.
    monthly_std : pd.Series
        Monthly climatological standard deviation, index = 1..12.
    """
    df = series.to_frame(name="val")
    df["month"] = df.index.month
    clim        = df.groupby("month")["val"].mean()
    monthly_std = df.groupby("month")["val"].std()
    anomaly = series - series.index.to_series().apply(lambda t: clim[t.month])
    if standardise:
        anomaly = anomaly / series.index.to_series().apply(lambda t: monthly_std[t.month])
    anomaly.name = series.name
    return anomaly, clim, monthly_std


def _contiguous_segments(series, min_len=30):
    """
    Return a list of contiguous non-NaN sub-series of minimum length min_len.

    Parameters
    ----------
    series : pd.Series
        Monthly timeseries that may contain NaN gaps.
    min_len : int
        Minimum number of valid observations a segment must have to be kept.

    Returns
    -------
    list of pd.Series
        Each element is a contiguous non-NaN segment.
    """
    valid_mask = series.notna()
    segments = []
    seg_start = None
    for i, (idx, ok) in enumerate(valid_mask.items()):
        if ok and seg_start is None:
            seg_start = idx
        elif not ok and seg_start is not None:
            seg = series[seg_start:series.index[i - 1]].dropna()
            if len(seg) >= min_len:
                segments.append(seg)
            seg_start = None
    # Close final segment
    if seg_start is not None:
        seg = series[seg_start:].dropna()
        if len(seg) >= min_len:
            segments.append(seg)
    return segments


def compute_acf_with_ci(series, max_lags=36, alpha=0.05):
    """
    Compute the empirical autocorrelation function (ACF) and approximate
    95 % confidence interval band (±1.96 / √N) for a complete series.

    Parameters
    ----------
    series : pd.Series
        Stationary monthly anomaly series. NaN values are dropped.
    max_lags : int
        Maximum lag (in months) to compute.
    alpha : float
        Significance level for the confidence interval (default 0.05).

    Returns
    -------
    lags : np.ndarray
        Lag indices 0, 1, …, max_lags.
    acf_vals : np.ndarray
        ACF values at each lag.
    ci_band : float
        Half-width of the symmetric 95 % CI (±ci_band).
    """
    valid = series.dropna().values
    acf_vals = sm_acf(valid, nlags=max_lags, fft=True)
    ci_band = 1.96 / np.sqrt(len(valid))
    lags = np.arange(max_lags + 1)
    return lags, acf_vals, ci_band


def compute_pacf_with_ci(series, max_lags=36, alpha=0.05):
    """
    Compute the empirical partial autocorrelation function (PACF) and
    approximate 95 % confidence interval band (±1.96 / √N).

    Parameters
    ----------
    series : pd.Series
        Stationary monthly anomaly series. NaN values are dropped.
    max_lags : int
        Maximum lag (in months) to compute. Must be < N/2.
    alpha : float
        Significance level for the confidence interval (default 0.05).

    Returns
    -------
    lags : np.ndarray
        Lag indices 0, 1, …, max_lags.
    pacf_vals : np.ndarray
        PACF values at each lag.
    ci_band : float
        Half-width of the symmetric 95 % CI (±ci_band).
    """
    valid = series.dropna().values
    max_lags = min(max_lags, len(valid) // 2 - 1)
    pacf_vals = sm_pacf(valid, nlags=max_lags, method="ywm")
    ci_band = 1.96 / np.sqrt(len(valid))
    lags = np.arange(max_lags + 1)
    return lags, pacf_vals, ci_band


def compute_acf_segmented(series, max_lags=36):
    """
    Compute ACF/PACF for a series that contains data gaps by processing each
    contiguous segment separately and returning the length-weighted average.

    This follows the lecturer's recommendation: compute ACF/PACF before and
    after each gap separately, then average them into a single estimate.

    Parameters
    ----------
    series : pd.Series
        Monthly timeseries with NaN gaps. DatetimeIndex required.
    max_lags : int
        Maximum lag (in months) to compute.

    Returns
    -------
    lags : np.ndarray
        Lag indices 0, 1, …, max_lags.
    acf_avg : np.ndarray
        Weighted-average ACF across segments.
    pacf_avg : np.ndarray
        Weighted-average PACF across segments.
    ci_band : float
        95 % CI half-width computed from total number of valid observations.
    segments_used : int
        Number of contiguous segments that were long enough to include.
    """
    segments = _contiguous_segments(series, min_len=max_lags + 10)
    if not segments:
        raise ValueError("No segment long enough to compute ACF/PACF.")

    acf_list, pacf_list, weights = [], [], []
    for seg in segments:
        n = len(seg)
        ml = min(max_lags, n // 2 - 1)
        a = sm_acf(seg.values, nlags=ml, fft=True)
        p = sm_pacf(seg.values, nlags=ml, method="ywm")
        # Pad to max_lags+1 with NaN if segment was short
        a_pad = np.full(max_lags + 1, np.nan)
        p_pad = np.full(max_lags + 1, np.nan)
        a_pad[:len(a)] = a
        p_pad[:len(p)] = p
        acf_list.append(a_pad)
        pacf_list.append(p_pad)
        weights.append(n)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    acf_stack  = np.array(acf_list)
    pacf_stack = np.array(pacf_list)

    acf_avg  = np.nansum(acf_stack  * weights[:, None], axis=0)
    pacf_avg = np.nansum(pacf_stack * weights[:, None], axis=0)

    total_n  = sum(len(s) for s in segments)
    ci_band  = 1.96 / np.sqrt(total_n)
    lags     = np.arange(max_lags + 1)

    return lags, acf_avg, pacf_avg, ci_band, len(segments)


def fit_ar(series, order):
    """
    Fit an AR(p) model to a stationary monthly anomaly series.

    Parameters
    ----------
    series : pd.Series
        Zero-mean, seasonally adjusted series. NaN values are dropped.
    order : int
        AR order p (number of autoregressive lags).

    Returns
    -------
    statsmodels ARIMAResults
        Fitted model result object. Key attributes: .params, .aic, .bic,
        .resid, .summary().
    """
    import warnings
    valid = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(valid, order=(order, 0, 0)).fit()
    return result


def fit_arma(series, p, q):
    """
    Fit an ARMA(p, q) model to a stationary monthly anomaly series.

    Parameters
    ----------
    series : pd.Series
        Zero-mean, seasonally adjusted series. NaN values are dropped.
    p : int
        Autoregressive order.
    q : int
        Moving-average order.

    Returns
    -------
    statsmodels ARIMAResults
        Fitted model result object. Key attributes: .params, .aic, .bic,
        .resid, .summary().
    """
    import warnings
    valid = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(valid, order=(p, 0, q)).fit()
    return result


def aic_grid_search(series, max_p=8, max_q=4):
    """
    Grid search over ARMA(p, q) orders using AIC and BIC as criteria.

    Only combinations with p + q ≤ 8 are evaluated to enforce parsimony
    (well below the lecturer's maximum of 12).

    Parameters
    ----------
    series : pd.Series
        Zero-mean, seasonally adjusted series. NaN values are dropped.
    max_p : int
        Maximum AR order to try (default 8).
    max_q : int
        Maximum MA order to try (default 4).

    Returns
    -------
    pd.DataFrame
        Columns: p, q, AIC, BIC — sorted by AIC ascending.
        The first row is the AIC-optimal order.
    """
    import warnings
    valid = series.dropna()
    rows = []
    for p in range(1, max_p + 1):
        for q in range(0, max_q + 1):
            if p + q > 8:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(valid, order=(p, 0, q)).fit()
                rows.append({"p": p, "q": q, "AIC": round(res.aic, 2), "BIC": round(res.bic, 2)})
            except Exception:
                pass
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


# =============================================================================
# SECTION 3 — Application & Evaluation
# =============================================================================

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox


def theoretical_acf(result, max_lags=36):
    """
    Compute the theoretical ACF implied by a fitted AR or ARMA model.

    Uses the AR and MA polynomial roots of the fitted statsmodels result
    to construct the corresponding ArmaProcess and compute its ACF.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted model result from fit_ar() or fit_arma().
    max_lags : int
        Number of lags to compute (default 36).

    Returns
    -------
    np.ndarray, shape (max_lags + 1,)
        Theoretical ACF values at lags 0, 1, …, max_lags.
    """
    ar = np.r_[1, -result.arparams]
    ma = np.r_[1,  result.maparams]
    process = ArmaProcess(ar, ma)
    return process.acf(max_lags + 1)


def residual_acf(result, max_lags=36):
    """
    Compute the empirical ACF of the model residuals.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted model result from fit_ar() or fit_arma().
    max_lags : int
        Maximum lag to compute (default 36).

    Returns
    -------
    lags : np.ndarray
        Lag indices 0, 1, …, max_lags.
    acf_vals : np.ndarray
        ACF of residuals at each lag.
    ci_band : float
        95% CI half-width (±1.96 / √N).
    """
    resid = result.resid.dropna().values
    acf_vals = sm_acf(resid, nlags=max_lags, fft=True)
    ci_band  = 1.96 / np.sqrt(len(resid))
    return np.arange(max_lags + 1), acf_vals, ci_band


def ljung_box_test(result, lags=20):
    """
    Portmanteau (Ljung-Box) test for residual independence.

    Tests H₀: residuals are white noise up to the given lag.
    Rejection (p < 0.05) indicates remaining autocorrelation.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted model result from fit_ar() or fit_arma().
    lags : int
        Number of lags to include in the test statistic (default 20).

    Returns
    -------
    dict with keys:
        lb_stat  (float) : Ljung-Box Q statistic.
        lb_pval  (float) : p-value of the test.
        reject   (bool)  : True if H₀ is rejected at 5% significance.
    """
    resid  = result.resid.dropna().values
    out    = acorr_ljungbox(resid, lags=[lags], return_df=True)
    stat   = float(out["lb_stat"].iloc[0])
    pval   = float(out["lb_pvalue"].iloc[0])
    return {"lb_stat": stat, "lb_pval": pval, "reject": pval < 0.05}


def ppcc_test(residuals, alpha=0.05):
    """
    Probability Plot Correlation Coefficient (PPCC) test for normality.

    Computes the correlation between the sorted residuals and the expected
    normal order statistics using Filliben's (1975) plotting positions.
    Rejects normality if the PPCC falls below the critical value at the
    given significance level.

    Parameters
    ----------
    residuals : array-like
        Residual series (NaN values are dropped).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        r        (float) : PPCC correlation coefficient.
        cv       (float) : Critical value at the given alpha.
        reject   (bool)  : True if normality is rejected (r < cv).
        n        (int)   : Number of residuals used.
    """
    x = np.sort(np.asarray(residuals, dtype=float))
    x = x[~np.isnan(x)]
    n = len(x)

    # Filliben plotting positions
    i  = np.arange(1, n + 1, dtype=float)
    m  = np.where(i == 1, 1 - 0.5 ** (1.0 / n),
         np.where(i == n, 0.5 ** (1.0 / n),
                  (i - 0.3175) / (n + 0.365)))
    z  = stats.norm.ppf(m)
    r, _ = stats.pearsonr(x, z)

    # Critical values (Filliben 1975, α = 0.05, one-sided)
    _ns  = np.array([10,  15,  20,  25,  30,  40,  50,
                     60,  75, 100, 150, 200, 300, 500])
    _cvs = np.array([0.9347, 0.9426, 0.9503, 0.9560, 0.9600,
                     0.9664, 0.9707, 0.9741, 0.9771, 0.9822,
                     0.9873, 0.9905, 0.9935, 0.9954])
    cv = float(np.interp(n, _ns, _cvs))

    return {"r": r, "cv": cv, "reject": r < cv, "n": n}

# =============================================================================
# SECTION 4 — Simulation
# =============================================================================

def simulate_from_model(result, n_steps=120, n_realizations=10, seed=42):
    """
    Generate synthetic realizations from a fitted AR or ARMA model.

    Draws innovation shocks from N(0, σ²) where σ² is the fitted residual
    variance, then filters them through the AR/MA polynomials using
    ArmaProcess.generate_sample.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted model from fit_ar() or fit_arma().
    n_steps : int
        Length of each synthetic series in months (default 120 = 10 years).
    n_realizations : int
        Number of independent realizations to generate (default 10).
    seed : int
        NumPy random seed for reproducibility (default 42).

    Returns
    -------
    np.ndarray, shape (n_realizations, n_steps)
        Each row is one independent synthetic realization.
    """
    np.random.seed(seed)
    ar      = np.r_[1, -result.arparams]
    ma      = np.r_[1,  result.maparams]
    sigma   = float(np.sqrt(result.params["sigma2"]))
    process = ArmaProcess(ar, ma)
    return np.vstack([
        process.generate_sample(nsample=n_steps, scale=sigma)
        for _ in range(n_realizations)
    ])


def sediment_mass(Q_monthly, C_monthly):
    """
    Compute monthly sediment mass flux M(t) = Q(t) · C(t) over the
    overlapping record of the two series.

    Unit derivation:
        Q [m³/s] × C [g/L] × (1000 L/m³) × (1 kg / 1000 g) = Q × C  [kg/s]
    The conversion factors cancel, so M [kg/s] = Q × C numerically.

    Parameters
    ----------
    Q_monthly : pd.Series
        Monthly discharge [m³/s] with DatetimeIndex.
    C_monthly : pd.Series
        Monthly SSC [g/L] with DatetimeIndex.

    Returns
    -------
    pd.Series
        Monthly sediment mass flux [kg/s] over the common non-NaN period.
    """
    common = Q_monthly.index.intersection(C_monthly.index)
    M = (Q_monthly.reindex(common) * C_monthly.reindex(common)).dropna()
    M.name = "M_kg_s"
    return M

# =============================================================================
# SECTION 5 — Independence Test
# =============================================================================

def correlation_test(x_series, y_series):
    """
    Compute Pearson and Spearman correlations between two monthly series
    over their common non-NaN overlap period.

    Parameters
    ----------
    x_series, y_series : pd.Series with DatetimeIndex

    Returns
    -------
    dict with keys:
        n           (int)   : number of paired observations
        pearson_r   (float) : Pearson r
        pearson_p   (float) : two-sided p-value for Pearson r
        spearman_r  (float) : Spearman ρ
        spearman_p  (float) : two-sided p-value for Spearman ρ
        x, y        (pd.Series) : aligned paired values used in the test
    """
    common = x_series.index.intersection(y_series.index)
    df = pd.DataFrame({"x": x_series.reindex(common),
                        "y": y_series.reindex(common)}).dropna()
    r_p, p_p = stats.pearsonr(df["x"], df["y"])
    r_s, p_s = stats.spearmanr(df["x"], df["y"])
    return {
        "n": len(df),
        "pearson_r": r_p, "pearson_p": p_p,
        "spearman_r": r_s, "spearman_p": p_s,
        "x": df["x"], "y": df["y"],
    }


def chi2_independence_test(x_series, y_series, n_bins=5):
    """
    Chi-squared test of independence between two monthly series using a
    quantile-based contingency table.

    Both series are discretised into n_bins equal-frequency bins; the
    resulting contingency table is tested with Pearson's chi-squared test.

    Parameters
    ----------
    x_series, y_series : pd.Series with DatetimeIndex
    n_bins : int
        Number of quantile bins (default 5, i.e. quintiles).

    Returns
    -------
    dict with keys:
        chi2    (float) : chi-squared test statistic
        p_value (float) : p-value (H0: x and y are independent)
        dof     (int)   : degrees of freedom
        reject  (bool)  : True if H0 rejected at 5% level
        n       (int)   : number of paired observations used
    """
    common = x_series.index.intersection(y_series.index)
    df = pd.DataFrame({"x": x_series.reindex(common),
                        "y": y_series.reindex(common)}).dropna()
    x_bins = pd.qcut(df["x"], n_bins, labels=False, duplicates="drop")
    y_bins = pd.qcut(df["y"], n_bins, labels=False, duplicates="drop")
    contingency = pd.crosstab(x_bins, y_bins)
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {"chi2": chi2, "p_value": p, "dof": dof,
            "reject": p < 0.05, "n": len(df)}
