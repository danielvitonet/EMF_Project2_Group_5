# -*- coding: utf-8 -*-
"""
Created on Sat May 17 20:01:43 2025

@author: HP
"""
###############################################################################
# EMPIRICAL METHODS IN FINANCE 2025
# =============================================================================
# GROUP MEMBERS:
# Daniel Vito Lobasso
# Thomas Nava
# Jacopo Sinigaglia
# Elvedin Muminovic
# =============================================================================
# Project #2: "Dynamic Allocation and VaR of a Portfolio"
# Goal: Highlight “volatility timing” and compute the Value-at-Risk of portfolio
#       when its return is described as a GARCH model or using extreme value theory.     
###############################################################################

# The code is optimized for Python 3.11.

###############################################################################

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from arch import arch_model
from datetime import datetime
from scipy.stats import kstest, norm, t, chi2, jarque_bera, gaussian_kde
from scipy.stats import genextreme
from scipy.optimize import minimize
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import lilliefors, acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF

###############################################################################
# PART 0: DIRECTORY AND DATA SET UP
###############################################################################

# Set working directory to the script's location
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())
# File paths
input_file = os.path.join("Data", "DATA_EMF.xlsx")

# LOAD AND PREPARE DATA:
def load_sheet(sheet_name, freq):
    # 1. Load the data from the Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name, parse_dates=['DATE'])
    
    # 2. Compute simple returns for stocks and bonds
    df['r_s'] = df['SWISS MARKET (SMI) - TOT RETURN IND'].pct_change()
    df['r_b'] = df['SW BENCHMARK 10 YEAR DS GOVT. INDEX - TOT RETURN IND'].pct_change()
    
    # 3. Compute weekly risk-free rate
    if freq == "weekly":
        df['r_f'] = df['SWISS FRANC S/T DEPO (FT/LSEG DS) - MIDDLE RATE'] / 100 / 52   # annual % -> weekly decimal
    elif freq == "daily":
        df['r_f'] = df['SWISS FRANC S/T DEPO (FT/LSEG DS) - MIDDLE RATE'] / 100 / 252  # annual % -> daily decimal
    
    out = df[['DATE','r_s','r_b','r_f']].dropna().set_index('DATE')
    return out

df_weekly = load_sheet('WEEKLY', 'weekly')
df_daily  = load_sheet('DAILY',  'daily')

#print("\n=== Weekly data (head) ===")
#print(df_weekly.head())
#print("\n=== Daily data (head) ===")
#print(df_daily.head())    

def plot_time_series(returns_df, title):
    plt.figure(figsize=(12,6))
    for col in returns_df.columns:
        plt.plot(returns_df.index, returns_df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_time_series(df_daily, "Time series returns")

###############################################################################
# PART 1: STATIC ALLOCATION
###############################################################################

df=df_weekly.copy()
# Sample means and covariance matrix
mu      = df[['r_s', 'r_b']].mean().values       # vector of sample means
Rf_bar  = df['r_f'].mean()                       # average risk-free rate
Sigma   = df[['r_s', 'r_b']].cov().values        # sample covariance matrix
print("\nSTATIC WEIGHTS INPUTS:")
print("\nsample mean vector")
print(mu)
print("\naverage risk-free rate")
print(Rf_bar)
print("\nsample covariance matrix")
print(Sigma)

# Compute optimal weights for λ = 2 and λ = 10
lambdas = [2, 10]
results = []
for lam in lambdas:
    #  α  =  (Σ⁻¹ (μ - Rf e)) / λ
    alpha = np.linalg.inv(Sigma).dot(mu - Rf_bar * np.ones(2)) / lam
    results.append({
        'lambda': lam,
        'alpha_s': alpha[0],
        'alpha_b': alpha[1],
        'alpha_cash': 1 - alpha.sum()
    })

# Display results
static_weights = pd.DataFrame(results).set_index('lambda')
print("\nOptimal weights for λ = 2 and λ = 10:")
print(static_weights)


###############################################################################
# PART 2: ESTIMATION OF A GARCH MODEL
###############################################################################

#df=df_weekly.copy()
# -----------------------------------------------------------------------------
# Q2.1: Non-normality (Lilliefors and K-S) and autocorrelation (Ljung-Box) on excess returns
# -----------------------------------------------------------------------------
df['ex_s'] = df['r_s'] - df['r_f']      # excess return stock
df['ex_b'] = df['r_b'] - df['r_f']      # excess return bond

plot_time_series(df, "Time series returns and excess returns")

print("\n= Q2.1: Non-normality & Autocorrelation of excess returns =\n")
for name in ['ex_s','ex_b']:
    x = df[name].dropna()
    
    print(f"\n= Kolmogorov-Smirnov test on {name}² =\n")
    s = x**2
    # Kolmogorov-Smirnov vs. N(0,1)
    z = (s - s.mean())/s.std(ddof=0)
    ks_stat, ks_p = kstest(z, 'norm')
    print(f"{name}²: K-S p={ks_p:.3f} {'(reject N₀)' if ks_p<0.05 else '(no rej)'}")
    
    print(f"\n= Kolmogorov-Smirnov test on {name} =\n")
    # Kolmogorov-Smirnov vs. N(0,1)
    z = (x - x.mean())/x.std(ddof=0)
    ks_stat, ks_p = kstest(z, 'norm')
    print(f"{name}: K-S p={ks_p:.3f} {'(reject N₀)' if ks_p<0.05 else '(no rej)'}")
    
    
    print(f"\n= Lilliefors test on {name} =\n")
    # Lilliefors test for Normal(μ,σ²)
    stat, pval = lilliefors(x, dist='norm')
    print(f"{name}: Lilliefors stat = {stat:.4f}, p = {pval:.4f}", 
          "(reject normality)" if pval < 0.05 else "(no rejection)")
    stat, pval = lilliefors(x**2, dist='norm')
    print(f"{name}²: Lilliefors stat = {stat:.4f}, p = {pval:.4f}", 
          "(reject normality)" if pval < 0.05 else "(no rejection)")
    
    
    # Approximate 5% Lilliefors critical value (large-n ≈0.089)
    c05 = 0.089
    # ECDF vs Normal CDF + Lilliefors band
    ecdf = ECDF(z)
    xs = np.linspace(z.min(), z.max(), 200)
    phi = norm.cdf(xs)
    plt.figure(figsize=(6,4))
    plt.step(ecdf.x, ecdf.y, where='post', label='Empirical CDF')
    plt.plot(xs, phi, 'k-', label='Normal CDF')
    plt.plot(xs, phi + c05, 'r--', label='5% c.v.') # label=f'±{c05:.3f} Lilliefors band'
    plt.plot(xs, phi - c05, 'r--')
    plt.title(f"{name}: ECDF vs Normal CDF\nLilliefors D={stat:.3f}, p={pval:.3f}")
    plt.xlabel('Standardized Excess Return')
    plt.ylabel('CDF')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Histogram vs Normal density
    plt.figure(figsize=(6,4))
    plt.hist(z, bins=30, density=True, alpha=0.6, edgecolor='black')
    xs2 = np.linspace(z.min(), z.max(), 200)
    plt.plot(xs2, norm.pdf(xs2), 'r--', label='N(0,1) density')
    plt.title(f"{name}: Histogram of Standardized Returns\nLilliefors D={stat:.3f}, p={pval:.3f}")
    plt.xlabel('Standardized Excess Return')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\n= Ljung-Box for {name} and {name}² (portmanteau test) =\n")
    # Ljung-Box on returns and squared returns (4 lags)
    lb_ret = acorr_ljungbox(x, lags=4, return_df=True)['lb_pvalue']
    lb_sq  = acorr_ljungbox(x**2, lags=4, return_df=True)['lb_pvalue']
    print(f"  Ljung-Box for {name} p={lb_ret.values} {'(autocorrelation)' if (lb_ret<0.05).any() else '(no autocorrelation)'}")
    print(f"  Ljung-Box for {name}² p={lb_sq.values} {'(volatility clustering - ARCH effects detected)' if (lb_sq<0.05).any() else '(no volatility clustering - no ARCH effects)'}\n")
    
    # equivalent to Ljung-Box on squared returns:
    print(f"\n= ARCH‐LM Test on {name} and {name}² =\n")
    # Test for ARCH effects in the residuals up to lag 4
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(x, nlags=4)
    print(f"  {name} ARCH‐LM (χ²) stat  = {lm_stat:.3f}, p‐value = {lm_pvalue:.4f}", 
          "(reject H0, variance is time‐varying)" if lm_pvalue < 0.05 else "(FAIL to reject H0, no ARCH detected, residuals’ variance is constant)")
    print(f"  {name} ARCH‐LM (F‐test) stat = {f_stat:.3f}, p‐value = {f_pvalue:.4f}\n")
    # H₀: No ARCH effects (i.e. residuals’ variance is constant).
    # H₁: ARCH effects present (variance is time‐varying).
    
    # ACF of returns
    plt.figure(figsize=(6,2.5))
    plot_acf(x, lags=4, title=f"ACF of {name}")
    plt.tight_layout()
    plt.show()
    
    # ACF of squared returns
    plt.figure(figsize=(6,2.5))
    plot_acf(x**2, lags=4, title=f"ACF of {name}^2")
    plt.tight_layout()
    plt.show()
    
from statsmodels.tsa.stattools import adfuller
# -----------------------------------------------------------------------------
# Q2.2: AR(1) on simple returns (Rs,t and Rb,t), extract residuals
# -----------------------------------------------------------------------------
print("\n=== Q2.2: AR(1) on Simple Returns ===\n")
ar1_res = {}
residuals = {}
for name in ['r_s','r_b']:
    y = df[name].dropna()
    X = sm.add_constant(y.shift(1)).dropna()
    y_aligned = y.loc[X.index]
    ar1 = sm.OLS(y_aligned, X).fit()
    ar1_res[name] = ar1
    print(f"{name} AR(1) results:\n", ar1.summary(), "\n")
    residuals[name] = ar1.resid
    
    print(adfuller(ar1.resid))
    
    # Test for ARCH effects in the residuals up to lag 4
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(ar1.resid, nlags=4)
    print(f"  ARCH‐LM (χ²) stat  = {lm_stat:.3f}, p‐value = {lm_pvalue:.4f}", 
          "(reject H0, variance is time‐varying)" if lm_pvalue < 0.05 else "(FAIL to reject H0, no ARCH detected, residuals’ variance is constant)")
    print(f"  ARCH‐LM (F‐test) stat = {f_stat:.3f}, p‐value = {f_pvalue:.4f}\n")
    # H₀: No ARCH effects (i.e. residuals’ variance is constant).
    # H₁: ARCH effects present (variance is time‐varying).
    

# -----------------------------------------------------------------------------
# Q2.3: GARCH(1,1) on AR(1) residuals via MLE (arch_model)
# -----------------------------------------------------------------------------

# BOOTSTRAP NON E' RICHIESTO DAL PROF #
def bootstrap_test(eps, name, B=500, scale=100, garch_res=None):
    """
    Perform LR bootstrap test for H0: alpha + beta = 1 with rescaled residuals,
    store results, and print the unconstrained GARCH summary.

    Parameters:
    -----------
    eps : np.ndarray
        AR(1) residuals (raw, unscaled).
    name : str
        Label for storing results (e.g., 'r_s', 'r_b').
    B : int
        Number of bootstrap replications.
    target_std_range : tuple
        Desired standard deviation range for rescaling.
    garch_res : dict
        Dictionary to store GARCH results.

    Returns:
    --------
    Updated garch_res dictionary with fitted models and bootstrap test results.
    """

    # Rescale residuals for numerical stability
    scale_factor = scale
    eps_scaled =  eps*scale_factor
    T = len(eps_scaled)
    
    if garch_res is None:
        garch_res = {}
    garch_res[name] = {}

    # Unconstrained estimation
    model_uncon = arch_model(eps_scaled, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
    res_uncon = model_uncon.fit(disp='off', cov_type='classic')
    loglik_uncon = res_uncon.loglikelihood

    print(f"\n--- {name} Unrestricted GARCH(1,1) Model Summary ---")
    print(res_uncon.summary())
    garch_res[name]['unrestricted'] = res_uncon

    # Constrained log-likelihood
    def constrained_loglik(params, data):
        omega, alpha = params
        beta = 1 - alpha
        if omega <= 0 or alpha < 0 or beta < 0 or alpha > 1 or beta > 1:
            return 1e6
        sigma2 = np.ones(len(data)) * np.var(data)
        for t in range(1, len(data)):
            sigma2[t] = omega + alpha * data[t - 1] ** 2 + beta * sigma2[t - 1]
        return -(-0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + data ** 2 / sigma2))

    # Estimate constrained model
    start_vals = [res_uncon.params['omega'], res_uncon.params['alpha[1]']]
    res_constr = minimize(constrained_loglik, start_vals, args=(eps_scaled,), method='L-BFGS-B')
    omega_c, alpha_c = res_constr.x
    loglik_constr = -res_constr.fun
    LR_obs = 2 * (loglik_uncon - loglik_constr)
    p_asymptotic = 1 - chi2.cdf(LR_obs, df=1)

    garch_res[name]['constrained'] = res_constr
    
    print(f" {name} Unrestricted GARCH MLE estimates under H1 (alpha + beta < 1):")
    ω, α, β = res_uncon.params[['omega','alpha[1]','beta[1]']]
    print(f" ω={ω:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}")
    
    beta_c = 1 - alpha_c  # since beta is implied by the constraint
    print(f" {name} Constrained GARCH MLE estimates under H0 (alpha + beta = 1):")
    print(f" ω={omega_c:.6f}, α={alpha_c:.6f}, β={beta_c:.6f}, α+β={(alpha_c+beta_c):.4f}")

    # Bootstrap DGP
    def simulate_series(omega, alpha, T, var_data):
        beta = 1 - alpha
        eps_sim = np.zeros(T)
        sigma2 = np.ones(T) * var_data
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps_sim[t - 1]**2 + beta * sigma2[t - 1]
            eps_sim[t] = np.sqrt(sigma2[t]) * np.random.normal()
        return eps_sim

    # Bootstrap loop
    bootstrap_LRs = []
    for _ in range(B):
        eps_b = simulate_series(omega_c, alpha_c, T, np.var(eps_scaled))
        try:
            model_b = arch_model(eps_b, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
            res_uncon_b = model_b.fit(disp='off')
            loglik_uncon_b = res_uncon_b.loglikelihood
            res_constr_b = minimize(constrained_loglik, start_vals, args=(eps_b,), method='L-BFGS-B')
            loglik_constr_b = -res_constr_b.fun
            LR_b = 2 * (loglik_uncon_b - loglik_constr_b)
            bootstrap_LRs.append(LR_b)
        except:
            continue

    p_bootstrap = np.mean(np.array(bootstrap_LRs) >= LR_obs)

    # Store test results
    garch_res[name]['bootstrap_test'] = {
        "scale factor": scale_factor,
        "LR observed": LR_obs,
        "p-asymptotic (χ²₁)": p_asymptotic,
        "p-bootstrap": p_bootstrap,
        "bootstrap iterations": B
        
    }

    return garch_res

#eps = residuals['r_s'].values
#garch_res = bootstrap_test(eps, name='SMI', B=500, scale=100)
#results = garch_res['SMI']['bootstrap_test']
#print("\n--- Bootstrap LR Test Results for H₀: α + β = 1 ---")
#print(f"{'Scale factor:':<30} {results['scale factor']}")
#print(f"{'Observed LR statistic:':<30} {results['LR observed']:.4f}")
#print(f"{'Asymptotic p-value (χ²₁):':<30} {results['p-asymptotic (χ²₁)']:.4f}")
#print(f"{'Bootstrap p-value:':<30} {results['p-bootstrap']:.4f}")
#print(f"{'Bootstrap replications:':<30} {results['bootstrap iterations']}")

#eps = residuals['r_b'].values
#garch_res = bootstrap_test(eps, name='SWISS GOVT. BONDS', B=500, scale=100, garch_res=garch_res)
#results = garch_res['SWISS GOVT. BONDS']['bootstrap_test']
#print("\n--- Bootstrap LR Test Results for H₀: α + β = 1 ---")
#print(f"{'Scale factor:':<30} {results['scale factor']}")
#print(f"{'Observed LR statistic:':<30} {results['LR observed']:.4f}")
#print(f"{'Asymptotic p-value (χ²₁):':<30} {results['p-asymptotic (χ²₁)']:.4f}")
#print(f"{'Bootstrap p-value:':<30} {results['p-bootstrap']:.4f}")
#print(f"{'Bootstrap replications:':<30} {results['bootstrap iterations']}")


print("\n=== Q2.3: GARCH(1,1) on AR(1) Residuals ===\n")
garch_res = {}
for name, eps in residuals.items():
    print(f"--- {name} GARCH(1,1) ---")
    model = arch_model(eps, mean='Zero', vol='GARCH', p=1, q=1, dist='normal', rescale=True)
    # print iteration info every iter, allow up to 1000 its
    res = model.fit(cov_type='classic', update_freq=10, disp='on', options={'maxiter':1000}) #cov_type='robust'
    garch_res[name] = res
    ω, α, β = res.params[['omega','alpha[1]','beta[1]']]
    print(res.summary())
    
    # Extract the scale factor c
    c = res.scale
    print(f"Applied scale factor: {c:.0f}\n")
    
    omega_unscaled = ω / (c**2)              # GARCH constant

    # Print un-scaled results
    print("\nUn-scaled (original data units):")
    print(f" ω={omega_unscaled:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}")
    
    #print("\nGARCH Model Results:")
    #print(f"ω={ω:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}")

    # Wald test H0: α+β = 1 vs H1: α+β < 1 (one‐sided)
    #    a) Compute difference and its variance via delta method
    d = (α + β) - 1.0
    cov_ab = res.param_cov.loc[['alpha[1]','beta[1]'],
                               ['alpha[1]','beta[1]']]
    var_d = (
    cov_ab.loc['alpha[1]','alpha[1]']
    + cov_ab.loc['beta[1]','beta[1]']
    + 2*cov_ab.loc['alpha[1]','beta[1]']
    )
    #    b) z‐statistic and one‐sided p‐value
    z_stat    = d / np.sqrt(var_d)
    p_one_sided = norm.cdf(z_stat)   # P(Z ≤ z_stat) for H1: α+β < 1
    print("\nWald test of H0: α+β = 1  vs  H1: α+β < 1")
    print(f" z = {z_stat:.4f}, one‐sided p‐value = {p_one_sided:.4f}\n")


# FATTO IN PIU' PER VEDERE COME CAMBIANO PARAMETRI #
print("\n=== Q2.3: AR(1)-GARCH(1,1) - NOT REQUESTED BY THE ASSIGNMENT, IT'S FOR COMPARISON ===\n")
#garch_res = {}
for name in ['r_s','r_b']:
    print(f"--- {name} GARCH(1,1) ---")
    model = arch_model(df[name], mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal', rescale=True)
    # print iteration info every iter, allow up to 1000 its
    res = model.fit(cov_type='classic', update_freq=10, disp='off', options={'maxiter':1000}) # cov_type='robust'
    #garch_res[name] = res
    
    # Extract the scale factor c
    c = res.scale
    print(f"Applied scale factor: {c:.0f}\n")
    
    ω, α, β = res.params[['omega','alpha[1]','beta[1]']]
    print(res.summary())
    
    mu_unscaled  = res.params['Const'] / c   # AR intercept
    omega_unscaled = ω / (c**2)              # GARCH constant

    # Print un-scaled results
    print("\nUn-scaled (original data units):")
    print(f" AR(1): const={mu_unscaled:.6f}, φ={res.params[f'{name}[1]']:.6f}")
    print(f" ω={omega_unscaled:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}")
    
    #print("\nAR - GARCH Model Results:")
    #print(f" AR(1): const={res.params['Const']:.6f}, φ={res.params[f'{name}[1]']:.6f}")
    #print(f" GARCH(1,1): ω={ω:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}")
    
    # Wald test H0: α+β = 1 vs H1: α+β < 1 (one‐sided)
    #    a) Compute difference and its variance via delta method
    d = (α + β) - 1.0
    cov_ab = res.param_cov.loc[['alpha[1]','beta[1]'],
                               ['alpha[1]','beta[1]']]
    var_d = (
    cov_ab.loc['alpha[1]','alpha[1]']
    + cov_ab.loc['beta[1]','beta[1]']
    + 2*cov_ab.loc['alpha[1]','beta[1]']
    )
    #    b) z‐statistic and one‐sided p‐value
    z_stat    = d / np.sqrt(var_d)
    p_one_sided = norm.cdf(z_stat)   # P(Z ≤ z_stat) for H1: α+β < 1
    print("\nWald test of H0: α+β = 1  vs  H1: α+β < 1")
    print(f" z = {z_stat:.4f}, one‐sided p‐value = {p_one_sided:.4f}\n")

# ----------------------------------------------------------------------------
# Wald test H0: α+β = 1 vs H1: α+β < 1 (one‐sided)
# Prof told me it is good as it is done here.
# TA: The constraint enforced by the arch package might affect the validity 
#     of the Wald test, since the boundary is excluded during the estimation.
# ----------------------------------------------------------------------------

garch_res['SMI'] = garch_res['r_s']
garch_res['SWISS GOVT. BONDS'] = garch_res['r_b']

# Volatility forecast (52 weeks)
horizon = 52
# 1) Unconditional variances
uc_s = ((garch_res['SMI'].params['omega'])/garch_res['SMI'].scale**2) / (1 - garch_res['SMI'].params['alpha[1]'] - garch_res['SMI'].params['beta[1]'])
uc_b = ((garch_res['SWISS GOVT. BONDS'].params['omega'])/garch_res['SWISS GOVT. BONDS'].scale**2) / (1 - garch_res['SWISS GOVT. BONDS'].params['alpha[1]'] - garch_res['SWISS GOVT. BONDS'].params['beta[1]'])
# 2) Conditional variance forecasts for next 52 weeks
vf_s = garch_res['SMI'].forecast(horizon=horizon, reindex=False).variance.iloc[0].values/garch_res['SMI'].scale**2
vf_b = garch_res['SWISS GOVT. BONDS'].forecast(horizon=horizon, reindex=False).variance.iloc[0].values/garch_res['SWISS GOVT. BONDS'].scale**2
# 3) Plotting
weeks = np.arange(1, horizon+1)
plt.figure(figsize=(12, 4))
# Stock volatility forecast
plt.subplot(1, 2, 1)
plt.plot(weeks, np.sqrt(vf_s) * 100, label='Conditional Vol')
plt.plot(weeks, np.sqrt(uc_s) * 100 * np.ones(horizon), '--', label='Unconditional Vol')
plt.title('SMI Weekly Volatility Forecast')
plt.xlabel('Weeks Ahead')
plt.ylabel('Weakly forcasted volatility in %')
plt.legend()
# Bond volatility forecast
plt.subplot(1, 2, 2)
plt.plot(weeks, np.sqrt(vf_b) * 100, label='Conditional Vol')
plt.plot(weeks, np.sqrt(uc_b) * 100 * np.ones(horizon), '--', label='Unconditional Vol')
plt.title('SWISS GOVT. BONDS Weekly Volatility Forecast')
plt.xlabel('Weeks Ahead')
plt.ylabel('Weakly forcasted volatility in %')
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# PART 3: DYNAMIC ALLOCATION
###############################################################################

df=df_weekly.copy()

#print("\n=== Q2.2: AR(1) on Simple Returns ===\n")
ar1_res = {}
residuals = {}
for name in ['r_s','r_b']:
    y = df[name].dropna()
    X = sm.add_constant(y.shift(1)).dropna()
    y_aligned = y.loc[X.index]
    ar1 = sm.OLS(y_aligned, X).fit()
    ar1_res[name] = ar1
    #print(f"{name} AR(1) results:\n", ar1.summary(), "\n")
    residuals[name] = ar1.resid

#print("\n=== Q2.3: GARCH(1,1) on AR(1) Residuals ===\n")
garch_res = {}
for name, eps in residuals.items():
    #print(f"--- {name} GARCH(1,1) ---")
    model = arch_model(eps, mean='Zero', vol='GARCH', p=1, q=1, dist='normal', rescale=True)
    # print iteration info every iter, allow up to 1000 its
    res = model.fit(cov_type='classic', update_freq=10, disp='off', options={'maxiter':1000}) #cov_type='robust'
    garch_res[name] = res

# --- 1. STATIC WEIGHTS (for comparison) ---
mu = df[['r_s', 'r_b']].mean().values
Rf_bar = df['r_f'].mean()
Sigma = df[['r_s', 'r_b']].cov().values

static_weights = {}
for lam in [2, 10]:
    alpha = np.linalg.inv(Sigma).dot(mu - Rf_bar * np.ones(2)) / lam
    static_weights[lam] = {
        'alpha_s': alpha[0],
        'alpha_b': alpha[1],
        'alpha_cash': 1 - alpha.sum()
    }

# --- 2. AR(1) FORECASTS AND GARCH CONDITIONAL VARIANCE ---
# Forecast means μ_{t+1|t} using AR(1)
a_s, rho_s = ar1_res['r_s'].params
a_b, rho_b = ar1_res['r_b'].params

mu_s = a_s + rho_s * df['r_s'] #.shift(1)  ho tolto lo shift
mu_b = a_b + rho_b * df['r_b'] #.shift(1)  ho tolto lo shift


# Residual correlation (assumed constant)
eps_s = ar1_res['r_s'].resid        # SMI AR(1) params
eps_b = ar1_res['r_b'].resid        # SWISS GOVT. BONDS AR(1) params
rho_sb = eps_s.corr(eps_b)

#index_s = eps_s.dropna().index
#index_b = eps_b.dropna().index

garch_res['SMI'] = garch_res['r_s']
garch_res['SWISS GOVT. BONDS'] = garch_res['r_b']

# GARCH conditional variance (1-step ahead forecast)
cond_s = pd.Series(
    garch_res['SMI'].conditional_volatility**2 / garch_res['SMI'].scale**2,
    #index=index_s
)
cond_b = pd.Series(
    garch_res['SWISS GOVT. BONDS'].conditional_volatility**2 / garch_res['SWISS GOVT. BONDS'].scale**2,
    #index=index_b
)
sigma2_s = cond_s.shift(-1) # cambiato da shift(1) a shift(-1) 
sigma2_b = cond_b.shift(-1) # cambiato da shift(1) a shift(-1) 
sigma_sb = rho_sb * np.sqrt(sigma2_s) * np.sqrt(sigma2_b)

#df_forecast = pd.concat([
#    mu_s, mu_b, sigma2_s, sigma2_b, sigma_sb, df['r_f']
#], axis=1, keys=['mu_s', 'mu_b', 'sigma2_s', 'sigma2_b', 'sigma_sb', 'r_f']).dropna()

#- APPORTATTO CORREZIONE: HO MESSO IL "GIUSTO" (almeno credo) Risk-free rate (uso il R_f del giorno in cui calcoliamo l'allocazione per il periodo successivo, prima usavo il R_f del girno in un cui avveniva la nuova allocazione)
df_forecast = pd.concat([
    mu_s, mu_b, sigma2_s, sigma2_b, sigma_sb, df['r_f'] #.shift(1) # Shift the risk-free rate by one day: use r_f[t - 1] instead of r_f[t]
], axis=1, keys=['mu_s', 'mu_b', 'sigma2_s', 'sigma2_b', 'sigma_sb', 'r_f']).dropna()

#rf = df_forecast['r_f'].loc[1]

# NOT REQUESTED BY ASSIGNMENT #
# PLOT DYNAMIC ANNUALIZED VOLATILITY #
# Annualize (weekly vol × sqrt(52))
#vol_stock_ann = garch_res['SMI'].conditional_volatility / garch_res['SMI'].scale  * np.sqrt(52)
#vol_bond_ann = garch_res['SWISS GOVT. BONDS'].conditional_volatility / garch_res['SWISS GOVT. BONDS'].scale * np.sqrt(52)
vol_stock_ann = np.sqrt(cond_s) * np.sqrt(52)
vol_bond_ann = np.sqrt(cond_b)  * np.sqrt(52)
# Create DataFrame
vol_df = pd.DataFrame({
    'Stock (SMI)': vol_stock_ann,
    'Bond (Swiss Gov)': vol_bond_ann
})
# Plot
plt.figure(figsize=(10, 6))
plt.plot(vol_df.index, vol_df['Stock (SMI)'], label='SMI')
plt.plot(vol_df.index, vol_df['Bond (Swiss Gov)'], label='SWISS GOVT. BONDS')
plt.title('Dynamic Annualized Volatility of Stocks and Bonds')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
##############################

# --- DYNAMIC WEIGHTS COMPUTATION ---
alphas_dyn = {}
for lam in [2, 10]:
    weights = []
    for t in df_forecast.index:
        μ = df_forecast.loc[t, ['mu_s', 'mu_b']].values
        Σ = np.array([[df_forecast.loc[t, 'sigma2_s'], df_forecast.loc[t, 'sigma_sb']],
                      [df_forecast.loc[t, 'sigma_sb'], df_forecast.loc[t, 'sigma2_b']]])
        #rf = df_forecast['r_f'].shift(1).loc[t] # 'correct' if I did't put shift(1) in df_forecast
        rf = df_forecast['r_f'].loc[t] # Now correctly lagged - shift put in df_forecast, so here is not necessary anymore
        alpha = np.linalg.inv(Σ).dot(μ - rf * np.ones(2)) / lam
        weights.append(alpha)
    alphas_dyn[lam] = pd.DataFrame(weights, columns=['alpha_s', 'alpha_b'], index=df_forecast.index)
    alphas_dyn[lam]['alpha_cash'] = 1 - alphas_dyn[lam].sum(axis=1)

# Create static weight time series
static_df = {}
for lam in [2, 10]:
    w = static_weights[lam]
    static_df[lam] = pd.DataFrame({
        'alpha_s': [w['alpha_s']] * len(df_forecast),
        'alpha_b': [w['alpha_b']] * len(df_forecast),
        'alpha_cash': [w['alpha_cash']] * len(df_forecast)
    }, index=df_forecast.index)

    
data_dir = os.path.join(base_dir, "Data") 
output_file = os.path.join(data_dir, "weights_output.xlsx")
# Check if file already exists
if not os.path.exists(output_file):
    with pd.ExcelWriter(output_file) as writer:
        alphas_dyn[2].to_excel(writer, sheet_name="dynamic_lambda2")
        alphas_dyn[10].to_excel(writer, sheet_name="dynamic_lambda10")
        static_df[2].to_excel(writer, sheet_name="static_lambda2")
        static_df[10].to_excel(writer, sheet_name="static_lambda10")
    print(f"Saved: {output_file}")
else:
    print(f"Skipped: {output_file} already exists.")



# --- CUMULATIVE RETURNS ---
CR, static_CR = {}, {}
non_log_CR, non_log_CR_static_CR = {}, {}
#AAAnon_log_CR = {}
for lam in [2, 10]:
    w_dyn = alphas_dyn[lam].copy()
    #Rf_dyn = df.loc[w_dyn.index, 'r_f']
    #Rs = df.loc[w_dyn.index, 'r_s']
    #Rb = df.loc[w_dyn.index, 'r_b']
    
    Rf_dyn = df.loc[w_dyn.index, 'r_f']
    Rs = df.loc[w_dyn.index, 'r_s'].shift(-1)
    Rb = df.loc[w_dyn.index, 'r_b'].shift(-1)
    
    
    # ABBASTANZA SICURO CHE NON VADA BENE:
    #Rp_dyn = (
    #w_dyn['alpha_s'] * Rs.shift(-1) +
    #w_dyn['alpha_b'] * Rb.shift(-1) +
    #w_dyn['alpha_cash'] * Rf_dyn.shift(-1)
    #)
    # USIAMO LA SEGUENTE...
    Rp_dyn = w_dyn['alpha_s'] * Rs + w_dyn['alpha_b'] * Rb + w_dyn['alpha_cash'] * Rf_dyn
    #non_log_CR[lam] = np.exp(np.log(1 + Rp_dyn).cumsum()) # NON LOGARITMICA
    non_log_CR[lam] = (1 + Rp_dyn).cumprod()
    CR[lam] = np.log(1 + Rp_dyn).cumsum()          # LOGARITMICA
    #AAAnon_log_CR[lam] = ((1 + Rp_dyn).cumprod()) # NON LOGARITMICA

    w_stat = static_weights[lam]
    Rp_stat = w_stat['alpha_s'] * Rs + w_stat['alpha_b'] * Rb + w_stat['alpha_cash'] * Rf_dyn
    #non_log_CR_static_CR[lam] = np.exp(np.log(1 + Rp_stat).cumsum()) # NON LOGARITMICA
    non_log_CR_static_CR[lam] = (1 + Rp_stat).cumprod() 
    static_CR[lam] = np.log(1 + Rp_stat).cumsum()          # LOGARITMICA

# PRINT Fial Cumulative (Log) Return
print('\n=== Fial Cumulative Log Return in %: ===\n')
print(f"Final Cumulative Log Return (in %) for dynamic λ=2: {(CR[2].iloc[-2]):.4%}")
print(f"Final Cumulative Log Return (in %) for static λ=2: {(static_CR[2].iloc[-2]):.4%}")
print(f"Final Cumulative Log Return (in %) for dynamic λ=2: {(CR[10].iloc[-2]):.4%}")
print(f"Final Cumulative Log Return (in %) for static λ=2: {(static_CR[10].iloc[-2]):.4%}")
print('\n=== Fial Cumulative Return in %: ===\n')
print(f"Final Cumulative Return (in %) for dynamic λ=2: {(non_log_CR[2].iloc[-2]-1):.4%}")
print(f"Final Cumulative Return (in %) for static λ=2: {(non_log_CR_static_CR[2].iloc[-2]-1):.4%}")
print(f"Final Cumulative Return (in %) for dynamic λ=2: {(non_log_CR[10].iloc[-2]-1):.4%}")
print(f"Final Cumulative Return (in %) for static λ=2: {(non_log_CR_static_CR[10].iloc[-2]-1):.4%}")



# PRINT Annualized Cumulative Log Return both in % and not :
print('\n=== Annualized Cumulative Log Return both in % and decimal: ===\n')
print('in %:')
print(f"Annualized Cumulative Log Return (in %) for dynamic λ=2: {(CR[2].iloc[-2] / 24):.4%}")
print(f"Annualized Cumulative Log Return (in %) for static λ=2: {(static_CR[2].iloc[-2] / 24):.4%}")
print(f"Annualized Cumulative Log Return (in %) for dynamic λ=10: {(CR[10].iloc[-2] / 24):.4%}")
print(f"Annualized Cumulative Log Return (in %) for static λ=10: {(static_CR[10].iloc[-2] / 24):.4%}")
print('in decimal:')
print(f"Annualized Cumulative Log Return for dynamic λ=2: {(CR[2].iloc[-2] / 24)}")
print(f"Annualized Cumulative Log Return for static λ=2: {(static_CR[2].iloc[-2] / 24)}")
print(f"Annualized Cumulative Log Return for dynamic λ=10: {(CR[10].iloc[-2] / 24)}")
print(f"Annualized Cumulative Log Return for static λ=10: {(static_CR[10].iloc[-2] / 24)}")

# PRINT Annualized Cumulative Return both in % and not :
print('\n=== Annualized Cumulative Return both in % and decimal: ===\n')
print('in %:')
print(f"Annualized Cumulative Return (in %) for dynamic λ=2: {((non_log_CR[2].iloc[-2]**(1/24)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for static λ=2: {((non_log_CR_static_CR[2].iloc[-2]**(1/24)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for dynamic λ=10: {((non_log_CR[10].iloc[-2]**(1/24)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for static λ=10: {((non_log_CR_static_CR[10].iloc[-2]**(1/24)-1)):.4%}")
print('in decimal:')
print(f"Annualized Cumulative Return for dynamic λ=2: {((non_log_CR[2].iloc[-2]**(1/24)-1))}")
print(f"Annualized Cumulative Return for static λ=2: {((non_log_CR_static_CR[2].iloc[-2]**(1/24)-1))}")
print(f"Annualized Cumulative Return for dynamic λ=10: {((non_log_CR[10].iloc[-2]**(1/24)-1))}")
print(f"Annualized Cumulative Return for static λ=10: {((non_log_CR_static_CR[10].iloc[-2]**(1/24)-1))}")
# using 52/1250 instead of 1/24 - maybe more accurate
print('in %:')
print(f"Annualized Cumulative Return (in %) for dynamic λ=2: {((non_log_CR[2].iloc[-2]**(52/1250)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for static λ=2: {((non_log_CR_static_CR[2].iloc[-2]**(52/1250)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for dynamic λ=10: {((non_log_CR[10].iloc[-2]**(52/1250)-1)):.4%}")
print(f"Annualized Cumulative Return (in %) for static λ=10: {((non_log_CR_static_CR[10].iloc[-2]**(52/1250)-1)):.4%}")
print('in decimal:')
print(f"Annualized Cumulative Return for dynamic λ=2: {((non_log_CR[2].iloc[-2]**(52/1250)-1))}")
print(f"Annualized Cumulative Return for static λ=2: {((non_log_CR_static_CR[2].iloc[-2]**(52/1250)-1))}")
print(f"Annualized Cumulative Return for dynamic λ=10: {((non_log_CR[10].iloc[-2]**(52/1250)-1))}")
print(f"Annualized Cumulative Return for static λ=10: {((non_log_CR_static_CR[10].iloc[-2]**(52/1250)-1))}")

# --- PLOT WEIGHTS ---
#alphas_dyn[2].index = alphas_dyn[lam].index - pd.Timedelta(weeks=1)
#alphas_dyn[10].index = alphas_dyn[lam].index - pd.Timedelta(weeks=1)
#fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#labels = ['Stock Weight', 'Bond Weight', 'Cash Weight']
#for i, name in enumerate(['alpha_s', 'alpha_b', 'alpha_cash']):
#    axs[i].plot(alphas_dyn[2].index, alphas_dyn[2][name], label='λ=2 (dynamic)', color='blue')
#    axs[i].plot(alphas_dyn[10].index, alphas_dyn[10][name], label='λ=10 (dynamic)', color='green')
#    axs[i].plot(static_df[2].index, static_df[2][name], '--', label='λ=2 (static)', color='red', alpha=0.6)
#    axs[i].plot(static_df[10].index, static_df[10][name], '--', label='λ=10 (static)', color='orange', alpha=0.6)
#    axs[i].set_ylabel(labels[i])
#    axs[i].grid(True)
#axs[2].set_xlabel("Date")
#axs[0].legend(ncol=2)
#fig.suptitle("Dynamic vs Static Portfolio Weights (λ = 2 and 10)")
#plt.tight_layout()
#plt.show()

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# mapping from column → human label
labels = {
    "alpha_s":    "Stock",
    "alpha_b":    "Bond",
    "alpha_cash": "Cash"
}

# choose a base (light) color for each leg
base_colors = {
    "alpha_s":    "tab:blue",
    "alpha_b":    "tab:orange",
    "alpha_cash": "tab:green"
}

for lam in [2, 10]:
    w_stat = static_df[lam].iloc[0]    # constant static weights
    dyn    = alphas_dyn[lam]           # dynamic weights DataFrame

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in ["alpha_s","alpha_b","alpha_cash"]:
        # get base RGB and a darker variant
        base_rgb = to_rgb(base_colors[col])
        dark_rgb = tuple(0.6 * c for c in base_rgb)

        # plot dynamic (dashed, thin, darker)
        ax.plot(
            dyn.index, dyn[col],
            #linestyle="--",
            linewidth=1,
            color=base_rgb,
            label=f"{labels[col]} (dynamic)"
        )

        # plot static (solid, thick, base color)
        ax.axhline(
            w_stat[col],
            linewidth=2.5,
            color=base_rgb,
            label=f"{labels[col]} (static)"
        )

    ax.set_title(f"Dynamic vs Static Allocation (λ = {lam})")
    ax.set_ylabel("Weight (1 = 100%)")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend(ncol=3, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.show()



#import matplotlib.pyplot as plt
#for lam in [2, 10]:
#    # pull out the constant static vector
#    w_stat = static_df[lam].iloc[0]
#    # dynamic DataFrame
#    dyn = alphas_dyn[lam]
#
#    fig, ax = plt.subplots(figsize=(12, 4))
#    # plot dynamic weights (dashed)
#    ax.plot(dyn.index, dyn['alpha_s'], '--', label='Stock (dynamic)',  color='C0')
#    ax.plot(dyn.index, dyn['alpha_b'], '--', label='Bond (dynamic)',   color='C1')
#    ax.plot(dyn.index, dyn['alpha_cash'], '--', label='Cash (dynamic)', color='C2')
#
#    # plot static horizontals (solid)
#    ax.axhline(w_stat['alpha_s'],   color='C0', label='Stock (static)')
#    ax.axhline(w_stat['alpha_b'],   color='C1', label='Bond (static)')
#    ax.axhline(w_stat['alpha_cash'],color='C2', label='Cash (static)')
#
#    ax.set_title(f"Dynamic vs Static Allocation (λ = {lam})")
#    ax.set_ylabel("Weight")
#    ax.set_xlabel("Date")
#    ax.legend(ncol=3, loc='upper left')
#    ax.grid(True)
#    plt.tight_layout()
#    plt.show()


# --- PLOT CUMULATIVE RETURNS ---
plt.figure(figsize=(12, 6))
plt.plot(non_log_CR[2], label='λ = 2 (dynamic)', color='blue')
plt.plot(non_log_CR[10], label='λ = 10 (dynamic)', color='green')
plt.plot(non_log_CR_static_CR[2], '--', label='λ = 2 (static)', color='red', alpha=0.6)
plt.plot(non_log_CR_static_CR[10], '--', label='λ = 10 (static)', color='orange', alpha=0.6)
plt.title('Cumulative Log Return: Dynamic vs Static Portfolio Allocation')
plt.xlabel('Date')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PLOT CUMULATIVE RETURNS ---
plt.figure(figsize=(12, 6))
plt.plot(CR[2], label='λ = 2 (dynamic)', color='blue')
plt.plot(CR[10], label='λ = 10 (dynamic)', color='green')
plt.plot(static_CR[2], '--', label='λ = 2 (static)', color='red', alpha=0.6)
plt.plot(static_CR[10], '--', label='λ = 10 (static)', color='orange', alpha=0.6)
plt.title('Cumulative Log Return: Dynamic vs Static Portfolio Allocation')
plt.xlabel('Date')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(CR[2], label='λ = 2 (dynamic)', color='blue')
plt.plot(static_CR[2], '--', label='λ = 2 (static)', color='red', alpha=0.6)
plt.title('Cumulative Log Return: Dynamic vs Static Portfolio Allocation (λ = 2)')
plt.xlabel('Date')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(CR[10], label='λ = 10 (dynamic)', color='green')
plt.plot(static_CR[10], '--', label='λ = 10 (static)', color='orange', alpha=0.6)
plt.title('Cumulative Log Return: Dynamic vs Static Portfolio Allocation (λ = 10)')
plt.xlabel('Date')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#delta_s = alphas_dyn[10]['alpha_s'].diff().abs()
#delta_b = alphas_dyn[10]['alpha_b'].diff().abs()
#f=1
#TC = ((delta_s + delta_b) * f).fillna(0)

# --- TRANSACTION COSTS ---
def compute_transaction_costs(weights_df, f):
    """
    Compute transaction costs based on absolute weight changes times fee f.
    The cash weight is ignored (no cost for risk-free asset).
    """
    delta_s = weights_df['alpha_s'].diff().abs()
    delta_b = weights_df['alpha_b'].diff().abs()
    TC = (delta_s + delta_b) * f
    return TC.fillna(0)

# Construct transaction cost range
f_values = np.linspace(0, 0.0003, 10000)  # from 0% to 0.03% - put 0.03% looking at the results to get closer approximation
break_even_f = {}

for lam in [2, 10]:
    # Shift index if not already corrected
    weights = alphas_dyn[lam].copy()
    
    #Rs = df['r_s'].shift(-1).loc[weights.index]
    #Rb = df['r_b'].shift(-1).loc[weights.index]
    #Rf_dyn = df['r_f'].shift(-1).loc[weights.index]
    #Rs = df['r_s'].loc[weights.index]
    #Rb = df['r_b'].loc[weights.index]
    #Rf_dyn = df['r_f'].loc[weights.index]
    Rf_dyn = df.loc[weights.index, 'r_f']
    Rs = df.loc[weights.index, 'r_s'].shift(-1)
    Rb = df.loc[weights.index, 'r_b'].shift(-1)

    # Dynamic portfolio return
    Rp_dyn = weights['alpha_s'] * Rs + weights['alpha_b'] * Rb + weights['alpha_cash'] * Rf_dyn

    # Static portfolio return
    w_stat = static_weights[lam]
    Rp_stat = w_stat['alpha_s'] * Rs + w_stat['alpha_b'] * Rb + w_stat['alpha_cash'] * Rf_dyn

    # Search for the f that equalizes cumulative returns
    for f in f_values:
        TC = compute_transaction_costs(weights, f)
        Rp_dyn_net = Rp_dyn - TC

        # Use log returns if explosive
        cum_log_dyn = np.log1p(Rp_dyn_net).cumsum()
        cum_log_stat = np.log1p(Rp_stat).cumsum()
        

        # Compare final wealth 
        if cum_log_dyn.iloc[-2] <= cum_log_stat.iloc[-2]:
        #if np.exp(cum_log_dyn.iloc[-2]) <= np.exp(cum_log_stat.iloc[-2]):
            break_even_f[lam] = f
            break

# --- DISPLAY RESULTS ---
for lam in [2, 10]:
    # a cost per unit of absolute change in portfolio weights, applied weekly based on the weekly turnover.
    #print('\n=== Cost per unit of absolute change in portfolio weights: ===\n')
    print(f"Break-even (unit) transaction cost rate for λ = {lam}: f = {break_even_f[lam]:.4%}")
    #print('\n=== "Annualized" cost per unit of absolute change in portfolio weights: ===\n')
    #print(f"'Annualized' break-even transaction cost rate for λ = {lam}: f = {(((1+break_even_f[lam])**52)-1):.4%}")

def compute_turnover (weights_df, f):
    """
    Compute turnover based on absolute weight changes.
    The cash weight is ignored (no cost for risk-free asset).
    """
    delta_s = weights_df['alpha_s'].diff().abs()
    delta_b = weights_df['alpha_b'].diff().abs()
    TU = (delta_s + delta_b)
    return TU.fillna(0)

for lam in [2, 10]:    
    if lam == 10 :
        weights = alphas_dyn[lam].copy()
        Rf_dyn = df.loc[weights.index, 'r_f']
        Rs = df.loc[weights.index, 'r_s'].shift(-1)
        Rb = df.loc[weights.index, 'r_b'].shift(-1)
        Rp_dyn = weights['alpha_s'] * Rs + weights['alpha_b'] * Rb + weights['alpha_cash'] * Rf_dyn
        w_stat = static_weights[lam]
        Rp_stat = w_stat['alpha_s'] * Rs + w_stat['alpha_b'] * Rb + w_stat['alpha_cash'] * Rf_dyn
        f = break_even_f[lam]
        TU = (compute_turnover(weights, f)).cumsum()
        last_TU = TU.iloc[-1]
        average_weekly_TU = last_TU / len(TU)
        average_annual_TU = last_TU/ 24
        TC = (compute_transaction_costs(weights, f)).cumsum()
        last_TC = TC.iloc[-1]
        average_weekly_TC = last_TC / len(TC)
        average_annual_TC = last_TC/ 24
        print('\n=== Summary for TC , Turnover, and f: ===\n')
        # Total Turnover and Transaction Costs 
        print(f"Total Turnover: {last_TU}")
        print(f"Total Transaction Costs: {last_TC}")
        # Annualized/Weekly cost impact
        print(f"Average Annual Turn Over: {average_annual_TU}")
        print(f"Average Annual Transaction cost: {average_annual_TC}")
        print(f"Average Weekly Turn Over: {average_weekly_TU}")
        print(f"Average Weekly Transaction cost: {average_weekly_TC}")
        
        #print(f"{((average_annual_TU)*f):.4%}")
        #print(f"{(average_annual_TC):.4%}")
        
        print(f"Annualized cost impact from break-even transaction cost rate for λ = {lam} is {(((1+break_even_f[lam]*average_weekly_TU)**52)-1):.4%}")
        print(f"Annualized cost impact from break-even transaction cost rate for λ = {lam} is {(((1+average_weekly_TC)**52)-1):.4%}")

# NON FIDATEVI DEL Break-even transaction cost rate for λ = 2: f = 0.0510%
# SPARA QUELLO SOLO PERCHE' E' IL MINIMO CHE HO MESSO. IN REALTA' NON ESISTE
# PERCHE' E' NEGATIVO GIA' IN PARTENZA

# We want to compare the performance gap vs transaction cost f
f_values = np.linspace(0.0001, 0.0003, 100)  # from 0% to 10%
# Store final cumulative wealths for each f and lambda
final_wealth_dyn = {2: [], 10: []}
final_wealth_stat = {}
for lam in [2, 10]:
    # Get weights and returns
    weights = alphas_dyn[lam].copy()
    #weights.index = weights.index - pd.Timedelta(weeks=1)  # align to decision date

    #Rs = df['r_s'].shift(-1).loc[weights.index]
    #Rb = df['r_b'].shift(-1).loc[weights.index]
    #Rf_dyn = df['r_f'].shift(-1).loc[weights.index]
    #Rs = df['r_s'].loc[weights.index]
    #Rb = df['r_b'].loc[weights.index]
    #Rf_dyn = df['r_f'].loc[weights.index]
    Rf_dyn = df.loc[weights.index, 'r_f']
    Rs = df.loc[weights.index, 'r_s'].shift(-1)
    Rb = df.loc[weights.index, 'r_b'].shift(-1)

    # Dynamic return pre-cost
    Rp_dyn = weights['alpha_s'] * Rs + weights['alpha_b'] * Rb + weights['alpha_cash'] * Rf_dyn

    # Static return
    w_stat = static_weights[lam]
    Rp_stat = w_stat['alpha_s'] * Rs + w_stat['alpha_b'] * Rb + w_stat['alpha_cash'] * Rf_dyn
    rp_stat = np.log1p(Rp_stat)
    #final_wealth_stat[lam] = np.exp(rp_stat.cumsum().iloc[-1])
    final_wealth_stat[lam] = rp_stat.cumsum().iloc[-2]

    # Evaluate net wealth for each f
    for f in f_values:
        #TC = (weights['alpha_s'].diff().abs() + weights['alpha_b'].diff().abs()) * f
        #TC = TC.fillna(0)
        #Rp_dyn_net = Rp_dyn - TC
        TC = compute_transaction_costs(weights, f)
        Rp_dyn_net = Rp_dyn - TC
        
        rp_dyn = np.log1p(Rp_dyn_net.mask(Rp_dyn_net <= -1))

        #Rp_dyn_net = np.where(Rp_dyn_net <= -1, np.nan, Rp_dyn_net)
        #rp_dyn = np.log1p(Rp_dyn_net)

        #final_wealth_dyn[lam].append(np.exp(rp_dyn.cumsum().iloc[-1]))
        final_wealth_dyn[lam].append(rp_dyn.cumsum().iloc[-2])

# Plotting
plt.figure(figsize=(10, 6))
for lam in [2, 10]:
    gap = np.array(final_wealth_dyn[lam]) - final_wealth_stat[lam]
    plt.plot(f_values * 100, gap, label=f'λ = {lam}')

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Performance Gap: Dynamic vs Static Allocation\nas a Function of Transaction Cost Rate (f)")
plt.xlabel("Transaction Cost Rate f (%)")
plt.ylabel("Final Wealth Difference (Dynamic - Static)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


    
###############################################################################
# PART 4: COMPUTING THE VaR OF A PORTFOLIO
###############################################################################

input_file = os.path.join("Data", "weights_output.xlsx")
weights  = pd.read_excel(input_file, sheet_name=None, parse_dates=['DATE'])

# Prepare weights for static and dynamic
alphas_dyn = {
    2: weights['dynamic_lambda2'],
    10: weights['dynamic_lambda10']
}
static_df = {
    2: weights['static_lambda2'],
    10: weights['static_lambda10']
}

for df in alphas_dyn.values():
    df.set_index("DATE", inplace=True)
for df in static_df.values():
    df.set_index("DATE", inplace=True)


# Prepare containers for returns and losses
#Rp = {}   # returns
Lp = {}   # losses

# Static portfolios: broadcast single weight vector to all days
for lam, wdf in static_df.items():
    w = wdf.iloc[0]   # constant α
    r = ( w.alpha_s   * df_daily["r_s"]
        + w.alpha_b   * df_daily["r_b"]
        + w.alpha_cash* df_daily["r_f"] )
    #Rp[f"static_{lam}"] = r
    Lp[f"static_{lam}"] = -r

# Dynamic portfolios: expand each weekly weight to its date + next 4 days
for lam, wdf in alphas_dyn.items():
    # build an all‐NA daily DataFrame
    dyn_daily = pd.DataFrame(index=df_daily.index,
                             columns=["alpha_s","alpha_b","alpha_cash"],
                             dtype=float)
    # fill each weekly block
    for dt, row in wdf.iterrows():
        pos = df_daily.index.get_loc(dt)
        days = df_daily.index[pos : pos+5]  # dt + next 4 trading days
        dyn_daily.loc[days] = row.values
    # drop leading gaps (before first dt)
    dyn_daily = dyn_daily.dropna()
    # compute returns only on filled days
    r = ( dyn_daily.alpha_s    * df_daily.loc[dyn_daily.index, "r_s"].shift(-5)
        + dyn_daily.alpha_b    * df_daily.loc[dyn_daily.index, "r_b"].shift(-5)
        + dyn_daily.alpha_cash * df_daily.loc[dyn_daily.index, "r_f"].shift(-5) ) # ADDED .shift(-1)
    
    #r = ( dyn_daily.alpha_s    * df_daily.loc[dyn_daily.index, "r_s"]
    #    + dyn_daily.alpha_b    * df_daily.loc[dyn_daily.index, "r_b"]
    #    + dyn_daily.alpha_cash * df_daily.loc[dyn_daily.index, "r_f"] )
    
    
    #Rp[f"dynamic_{lam}"] = r
    Lp[f"dynamic_{lam}"] = -r

# 7) (Optional) bundle into DataFrames
#df_Rp = pd.DataFrame(Rp)
df_Lp = pd.DataFrame(Lp)

#df_Lp = df_Lp.iloc[9:]  # removes the first 12 rows from all columns
df_Lp = df_Lp.dropna()

# 8) Inspect / save
#print(df_Lp.head())

# Compute Q4.1 results: deduced quantile θ = 99% and Unconditional VaR
z_99 = norm.ppf(0.99)  # 99% quantile of standard normal , IS THIS RIGHT??? DUSCUSSION !!!
results_q41 = {
    key: {
        'mean_loss': losses.mean(),
        'variance_loss': losses.var(),
        'VaR_99': losses.mean() + z_99 * losses.std()
    }
    for key, losses in Lp.items()
}

#print(results_q41)
# Convert to DataFrame for pretty display FOR Q4.1
df_q41 = pd.DataFrame(results_q41).T
df_q41.index.name = 'Portfolio'
df_q41.columns = ['Mean Loss', 'Variance of Loss', 'VaR (99%)']
print("\n=== Q4.1: estimated UNCONDITIONAL: Mean, Variance, and VaR (99%) ===\n")
print(df_q41)




# Q4.2 , Q4.3, Q4.4, Q4.5 :

# PROBLEMS WITH THIS SECTION:
# - CALCULATIONS OF CONDITIONAL QUARTILE IN Q4.2 AND ITS TEMPORAL EVOLUTION (HOW DO WE COMPUTE IT?)
# - CALCULATIONS OF QUARTILE IN Q4.5 AND ITS TEMPORAL EVOLUTION (HOW DO WE COMPUTE IT?)

VaR_99 = {}          # Q4.2 Time evolution of 99% Conditional VaR (GARCH)
gev_params_dfLp={}   # Q4.3 Parameters (ξ, ϖ, ψ)’ of the GEV 
q99_results_dfLp={}  # Q4.4 Computed 99% quantile of m_τ distribution and Deduced 99% quantile of z_hat distribution
VaR_99_GEV_dfLp = {} # Q4.5 Temporal evolution of the 99% GEV VaR 

qq99_results_dfLp = {} # prova
#qqVaR_99_GEV_dfLp = {} # prova


print("\n=== Q4.2: AR(1)-GARCH(1,1) ===\n")
garch_res_loss = {}
for name in ['static_2', 'static_10', 'dynamic_2','dynamic_10']:
    print(f"--- {name} GARCH(1,1) ---")
    am = arch_model(df_Lp[name], mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal', rescale=True)
    # print iteration info every iter, allow up to 1000 its
    res = am.fit(cov_type='classic', update_freq=10, disp='off', options={'maxiter':1000}) # cov_type='robust'
    #print(res.summary())
    garch_res_loss[name] = res
    
    # Extract scale factor to adjust back to original units
    scale = res.scale

    # Conditional mean: μ_{t+1} = a + ρ * L_t
    params = res.params
    a = params['Const'] /scale # UN-SCALE
    rho = params[f'{name}[1]']
    L_lagged = df_Lp[name].shift(1)
    mu_t = a + rho * L_lagged

    # Conditional standard deviation (scaled back)
    sigma_t = res.conditional_volatility / np.sqrt(scale)

    # Time-varying 99% 1-day VaR
    var_series = mu_t + z_99 * sigma_t
    VaR_99[name] = var_series # Q4.2
    
    ####### Q4.3 - Q4.4

    #Lp = df_Lp[name]
    #mu = Lp.mean()
    #sigma = Lp.std()
    #z_hat = (Lp - mu) / sigma
    
    
    Lp = df_Lp[name]
    # Conditional standardized residuals
    z_hat = (Lp - mu_t) / sigma_t
        
    # Q4.3: GEV estimation
    block_size = 60
    T_valid = len(z_hat.dropna())
    n_blocks = T_valid // block_size
    m_tau = [z_hat.dropna().iloc[i * block_size:(i + 1) * block_size].max() for i in range(n_blocks)]
    m_tau = np.array(m_tau)

    # Estimate GEV parameters
    shape, loc, scale_gev = genextreme.fit(m_tau)
    gev_params_dfLp[name] = {'shape (ξ)': -shape, 'location (ω)': loc, 'scale (ψ)': scale_gev}
    
     
    # Q4.4: Quantile of maxima
    q99_m = loc + (scale_gev / -shape) * ((-np.log(0.99))**(shape) - 1)
    # or
    qq99_m = genextreme.ppf(0.99, shape, loc=loc, scale=scale_gev)

    # Quantile of return distribution (standardized residuals) | 99% quantile of z_hat
    N = block_size
    q99_z_hat = loc + (scale_gev / -shape) * ((-N * np.log(0.99))**(shape) - 1)
    # or - data-based (non-parametric quantile estimator)
    #tail_prob = 1 - 0.99**(1/60)
    #qq99_z_hat = np.quantile(z_hat.dropna(), 1 - tail_prob) 

    q99_results_dfLp[name] = {
        '99% quantile of maxima m_τ': q99_m,
        '99% quantile of z_hat': q99_z_hat
    }
    
    # PROVA da non usare !
    qq99_results_dfLp[name] = {
        '99% quantile of maxima m_τ': qq99_m,
        #'99% quantile of z_hat': qq99_z_hat
    }
    
    # PROVA
    #qqVaR_99_GEV = mu_t + qq99_z_hat * sigma_t
    #qqVaR_99_GEV.name = f'qqVaR_99_GEV_{name}'
    #qqVaR_99_GEV_dfLp[name] = qqVaR_99_GEV
    

    # Q4.5: VaR from GEV quantile
    VaR_99_GEV = mu_t + q99_z_hat * sigma_t
    VaR_99_GEV.name = f'VaR_99_GEV_{name}'
    VaR_99_GEV_dfLp[name] = VaR_99_GEV
############################################
    
    # Step 1: Divide into 60-day blocks and get maxima
    #block_size = 60
    #T = len(z_hat)
    #n_blocks = T // block_size
    #m_tau = [z_hat.iloc[i * block_size:(i + 1) * block_size].max() for i in range(n_blocks)]
    #m_tau = np.array(m_tau)

    # Step 2: Fit GEV distribution #Q4.3
    #shape, loc, scale = genextreme.fit(m_tau)
    #gev_params_dfLp[name] = {'shape (ξ)': shape, 'location (ω)': loc, 'scale (ψ)': scale}
    
    # Q4.4
    # Step 3: Compute 99% quantile of maxima distribution
    #q99_m = genextreme.ppf(0.99, shape, loc=loc, scale=scale)
    # Step 4: Deduce 99% quantile of z_hat
    #tail_prob = 1 - 0.99**(1/60) # CONTROLLARE FORMULA DALLE SLIDES
    #q99_z_hat = np.quantile(z_hat, 1 - tail_prob)
    #gev_params_dfLp[name] = {'shape (ξ)': shape, 'location (ω)': loc, 'scale (ψ)': scale}
    #q99_results_dfLp[name] = {'99% quantile of maxima m_τ': q99_m, '99% quantile of z_hat': q99_z_hat}

    ### Q4.5 CREDO SIA IL 4.5, CONTROLLARE!!! 
    #VaR_99_GEV = mu_t + q99_z_hat * sigma_t
    #VaR_99_GEV.name = f'VaR_99_GEV_{name}'
    #VaR_99_GEV_dfLp[name] = VaR_99_GEV

    # Return the head of the series
    #VaR_99_GEV.head()

# Convert each dictionary to a nicely formatted DataFrame for display
gev_params_df = pd.DataFrame(gev_params_dfLp).T
q99_results_df = pd.DataFrame(q99_results_dfLp).T
var_99_summary = pd.DataFrame({k: v.describe() for k, v in VaR_99.items()}).T[['mean', 'std', 'min', 'max']]
var_99_gev_summary = pd.DataFrame({k: v.describe() for k, v in VaR_99_GEV_dfLp.items()}).T[['mean', 'std', 'min', 'max']]

# PROVA
qq99_results_df = pd.DataFrame(qq99_results_dfLp).T
#qqvar_99_gev_summary = pd.DataFrame({k: v.describe() for k, v in qqVaR_99_GEV_dfLp.items()}).T[['mean', 'std', 'min', 'max']]



print("\n=== Q4.2: Time evolution of 99% Conditional (GARCH) VaR - look plot ===\n")
print(var_99_summary)
# Tables for Q4.3 & Q4.4 (GEV parameters and quantiles)
print("\n=== Q4.3: GEV parameters ===\n")
print(gev_params_df)
print("\n=== Q4.4: GEV quantiles ===\n")
print(q99_results_df)
print(qq99_results_df)

print("\n=== Q4.5: Time evolution of 99% GEV VaR - look plot ===\n")
print(var_99_gev_summary)
#print(qqvar_99_gev_summary) 


import matplotlib.pyplot as plt 

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["figure.figsize"] = (14, 6)

# Dictionary for clean labels
labels = {
    'static_2': 'Static (λ=2)',
    'dynamic_2': 'Dynamic (λ=2)',
    'static_10': 'Static (λ=10)',
    'dynamic_10': 'Dynamic (λ=10)',
}

# Plot 1: GARCH VaR for λ=2
plt.figure()
plt.plot(VaR_99['dynamic_2'], label=labels['dynamic_2'], color='#ff7f0e')
plt.plot(VaR_99['static_2'], label=labels['static_2'], color='#1f77b4')
plt.title('Temporal Evolution of 99% VaR (GARCH) - λ=2')
plt.xlabel('Date')
plt.ylabel('VaR')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: GARCH VaR for λ=10
plt.figure()
plt.plot(VaR_99['dynamic_10'], label=labels['dynamic_10'], color='#ff7f0e')
plt.plot(VaR_99['static_10'], label=labels['static_10'], color='#1f77b4')
plt.title('Temporal Evolution of 99% VaR (GARCH) - λ=10')
plt.xlabel('Date')
plt.ylabel('VaR')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: GEV VaR for λ=2
plt.figure()
plt.plot(VaR_99_GEV_dfLp['dynamic_2'], label=labels['dynamic_2'], color='#ff7f0e')
plt.plot(VaR_99_GEV_dfLp['static_2'], label=labels['static_2'], color='#1f77b4')
plt.title('Temporal Evolution of 99% VaR (GEV) - λ=2')
plt.xlabel('Date')
plt.ylabel('VaR')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: GEV VaR for λ=10
plt.figure()
plt.plot(VaR_99_GEV_dfLp['dynamic_10'], label=labels['dynamic_10'], color='#ff7f0e')
plt.plot(VaR_99_GEV_dfLp['static_10'], label=labels['static_10'], color='#1f77b4')
plt.title('Temporal Evolution of 99% VaR (GEV) - λ=10')
plt.xlabel('Date')
plt.ylabel('VaR')
plt.legend()
plt.tight_layout()
plt.show()







#import matplotlib.pyplot as plt
#
# Setup
#plt.style.use('seaborn-v0_8-whitegrid')
#plt.rcParams["figure.figsize"] = (14, 6)
#
## Dictionary for clean labels
#labels = {
#    'static_2': 'Static (λ=2)',
#    'dynamic_2': 'Dynamic (λ=2)',
#    'static_10': 'Static (λ=10)',
#    'dynamic_10': 'Dynamic (λ=10)',
#}
#
# Plot 1: GARCH VaR for λ=2
#plt.figure()
#plt.plot(VaR_99['dynamic_2'], label='Dynamic (λ=2)')
#plt.plot(VaR_99['static_2'], label='Static (λ=2)')
#plt.title('Temporal Evolution of 99% VaR (GARCH) - λ=2')
#plt.xlabel('Date')
#plt.ylabel('VaR')
#plt.legend()
#plt.tight_layout()
#plt.show()
#
# Plot 2: GARCH VaR for λ=10
#plt.figure()
#plt.plot(VaR_99['dynamic_10'], label='Dynamic (λ=10)')
#plt.plot(VaR_99['static_10'], label='Static (λ=10)')
#plt.title('Temporal Evolution of 99% VaR (GARCH) - λ=10')
#plt.xlabel('Date')
#plt.ylabel('VaR')
#plt.legend()
#plt.tight_layout()
#plt.show()
#
# Plot 3: GEV VaR for λ=2
#plt.figure()
#plt.plot(VaR_99_GEV_dfLp['dynamic_2'], label='Dynamic (λ=2)')
#plt.plot(VaR_99_GEV_dfLp['static_2'], label='Static (λ=2)')
#plt.title('Temporal Evolution of 99% VaR (GEV) - λ=2')
#plt.xlabel('Date')
#plt.ylabel('VaR')
#plt.legend()
#plt.tight_layout()
#plt.show()
#
# Plot 4: GEV VaR for λ=10
#plt.figure()
#plt.plot(VaR_99_GEV_dfLp['dynamic_10'], label='Dynamic (λ=10)')
#plt.plot(VaR_99_GEV_dfLp['static_10'], label='Static (λ=10)')
#plt.title('Temporal Evolution of 99% VaR (GEV) - λ=10')
#plt.xlabel('Date')
#plt.ylabel('VaR')
#plt.legend()
#plt.tight_layout()
#plt.show()

# Setup for consistent visuals
#sns.set(style="whitegrid")
#plt.rcParams['figure.figsize'] = (14, 6)

# Plot VaR evolution (Q4.2 and Q4.5)
#fig, ax = plt.subplots()
#for name in ['static_2', 'static_10', 'dynamic_2', 'dynamic_10']:
#    if name in VaR_99 and name in VaR_99_GEV_dfLp:
#        ax.plot(VaR_99[name], label=f'VaR-GARCH {name}', linestyle='--')
#        ax.plot(VaR_99_GEV_dfLp[name], label=f'VaR-GEV {name}', alpha=0.8)
#ax.set_title("Q4.2 & Q4.5: Temporal Evolution of 99% VaR (GARCH vs GEV)")
#ax.set_ylabel("VaR")
#ax.set_xlabel("Date")
#ax.legend()
#plt.tight_layout()
#plt.show()


#import matplotlib.pyplot as plt
#import seaborn as sns

## Setup for consistent visuals
#sns.set(style="whitegrid")
#plt.rcParams['figure.figsize'] = (14, 6)

# Plot separate graphs for each portfolio's VaR (GARCH vs GEV)
#for name in ['static_2', 'static_10', 'dynamic_2', 'dynamic_10']:
#    if name in VaR_99 and name in VaR_99_GEV_dfLp:
#        plt.figure()        
#        plt.plot(VaR_99_GEV_dfLp[name], label='VaR-GEV', alpha=0.8)
#        plt.plot(VaR_99[name], label='VaR-GARCH') #, linestyle='--'
#        plt.title(f"Q4.2 & Q4.5: 99% VaR Evolution for {name}")
#        plt.ylabel("VaR")
#        plt.xlabel("Date")
#        plt.legend()
#        plt.tight_layout()
#        plt.show()






#plt.figure(figsize=(12, 6))
#for name in ['static_2', 'static_10', 'dynamic_2', 'dynamic_10']:
#    plt.plot(df_Lp[f'VaR_99_{name}'], label=f'VaR 99% - {name}')
#
#plt.title('Time Evolution of Conditional 99% VaR (AR(1)-GARCH(1,1))')
#plt.xlabel('Date')
#plt.ylabel('VaR')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

#from scipy.stats import genextreme

#gev_params_dfLp={} 
#q99_results_dfLp={}
#for name in ['static_2', 'static_10', 'dynamic_2','dynamic_10']:
#    Lp = df_Lp[name]
#    mu = Lp.mean()
#    sigma = Lp.std()
#    z_hat = (Lp - mu) / sigma
#
#    # Step 1: Divide into 60-day blocks and get maxima
#    block_size = 60
#    T = len(z_hat)
#    n_blocks = T // block_size
#    m_tau = [z_hat.iloc[i * block_size:(i + 1) * block_size].max() for i in range(n_blocks)]
#    m_tau = np.array(m_tau)
#
#    # Step 2: Fit GEV distribution
#    shape, loc, scale = genextreme.fit(m_tau)
#
#    # Step 3: Compute 99% quantile of maxima distribution
#    q99_m = genextreme.ppf(0.99, shape, loc=loc, scale=scale)
#
#    # Step 4: Deduce 99% quantile of z_hat
#    tail_prob = 1 - 0.99**(1/60) # CONTROLLARE FORMULA DALLE SLIDES
#    q99_z_hat = np.quantile(z_hat, 1 - tail_prob)
#
#    gev_params_dfLp[name] = {'shape (ξ)': shape, 'location (ω)': loc, 'scale (ψ)': scale}
#    q99_results_dfLp[name] = {'99% quantile of maxima m_τ': q99_m, '99% quantile of z_hat': q99_z_hat}
#
#    #gev_params_dfLp, q99_results_dfLp


