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
import pandas as pd
import numpy as np
from scipy.stats import kstest, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.diagnostic import het_arch
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import minimize
from scipy.stats import chi2
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, norm, gaussian_kde
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS, add_constant

###############################################################################
# PART 0: DIRECTORY AND DATA SET UP
###############################################################################

# Set working directory to the script's location
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())
# File paths
input_file = os.path.join("Data", "WEAKLY_DATA_EMF.xlsx")

# LOAD AND PREPARE DATA: 
# 1. Load the data from the Excel file 
smi  = pd.read_excel(input_file, sheet_name='SMI',  parse_dates=['DATE'])
bond = pd.read_excel(input_file, sheet_name='Bond', parse_dates=['DATE'])
rate = pd.read_excel(input_file, sheet_name='Rate', parse_dates=['DATE'])
# 2. Compute simple weekly returns for stocks and bonds
smi['r_s'] = smi['SWISS MARKET (SMI) - TOT RETURN IND'].pct_change()
bond['r_b'] = bond['SW BENCHMARK 10 YEAR DS GOVT. INDEX - TOT RETURN IND'].pct_change()
# 3. Compute weekly risk-free rate (take % and divide by 52)
rate['r_f'] = rate['SWISS FRANC S/T DEPO (FT/LSEG DS) - MIDDLE RATE'] / 52
# 4. Merge into a single DataFrame
df = (
    smi[['DATE', 'r_s']]
    .merge(bond[['DATE', 'r_b']], on='DATE')
    .merge(rate[['DATE', 'r_f']], on='DATE')
    .dropna()
    .set_index('DATE')
)

#print("\nFirst few rows of merged data:")
#print(df.head())


###############################################################################
# PART 1: STATIC ALLOCATION
###############################################################################

# Sample means and covariance matrix
mu      = df[['r_s', 'r_b']].mean().values       # vector of sample means
Rf_bar  = df['r_f'].mean()                       # average risk-free rate
Sigma   = df[['r_s', 'r_b']].cov().values        # sample covariance matrix

print(mu)
print(Rf_bar)
print(Sigma)

# Compute optimal weights for λ = 2 and λ = 10
lambdas = [2, 10]
results = []
for lam in lambdas:
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

# -----------------------------------------------------------------------------
# Q2.1: Non-normality (K-S) and autocorrelation (Ljung-Box) on excess returns
# -----------------------------------------------------------------------------
df['ex_s'] = df['r_s'] - df['r_f']      # excess return stock
df['ex_b'] = df['r_b'] - df['r_f']      # excess return bond

print("\n=== Q2.1: Non-normality & Autocorrelation of Excess Returns ===\n")
for name in ['ex_s','ex_b']:
    x = df[name].dropna()
    # Kolmogorov-Smirnov vs. N(0,1)
    z = (x - x.mean())/x.std(ddof=0)
    ks_stat, ks_p = kstest(z, 'norm')
    print(f"{name}: K-S p={ks_p:.3f} {'(reject N₀)' if ks_p<0.05 else '(no rej)'}")
    
    # Lilliefors test for Normal(μ,σ²)
    stat, pval = lilliefors(x, dist='norm')
    print(f"{name}: Lilliefors stat = {stat:.4f}, p = {pval:.4f}", 
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
    plt.plot(xs, phi + c05, 'r--', label=f'±{c05:.3f} Lilliefors band')
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
    
    # Ljung-Box on returns and squared returns (4 lags)
    lb_ret = acorr_ljungbox(x, lags=4, return_df=True)['lb_pvalue']
    lb_sq  = acorr_ljungbox(x**2, lags=4, return_df=True)['lb_pvalue']
    print(f"  LB returns p={lb_ret.values} {'(AC)' if (lb_ret<0.05).any() else '(no AC)'}")
    print(f"  LB squared p={lb_sq.values} {'(clustering)' if (lb_sq<0.05).any() else '(no clustering)'}\n")
    
    # ACF of returns
    plt.figure(figsize=(6,2.5))
    plot_acf(x, lags=4, title=f"{name}: ACF(Returns)")
    plt.tight_layout()
    plt.show()
    
    # ACF of squared returns
    plt.figure(figsize=(6,2.5))
    plot_acf(x**2, lags=4, title=f"{name}: ACF(Squared Returns)")
    plt.tight_layout()
    plt.show()
    
    
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
    print(f"{name} AR(1): params={ar1.params.round(4).to_dict()}, R²={ar1.rsquared:.3f}")
    #resid = ar1.resid
    #residuals[name] = resid
    residuals[name] = ar1.resid

print("\n=== ARCH‐LM Test on AR(1) Residuals ===\n")
for name, resid in residuals.items():
    # Test for ARCH effects in the residuals up to lag 4
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=4)
    
    print(f"{name}:")
    print(f"  ARCH‐LM (χ²) stat  = {lm_stat:.3f}, p‐value = {lm_pvalue:.4f}", 
          "(reject H0, variance is time‐varying)" if lm_pvalue < 0.05 else "(FAIL to reject H0, no ARCH detected, residuals’ variance is constant)")
    print(f"  ARCH‐LM (F‐test) stat = {f_stat:.3f}, p‐value = {f_pvalue:.4f}\n")

    # H₀: No ARCH effects (i.e. residuals’ variance is constant).
    # H₁: ARCH effects present (variance is time‐varying).

# -----------------------------------------------------------------------------
# Q2.3: GARCH(1,1) on AR(1) residuals via MLE (arch_model)
# -----------------------------------------------------------------------------
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


print("\n=== Q2.3: AR(1)-GARCH(1,1) ===\n")
garch_res = {}
for name in ['r_s','r_b']:
    print(f"--- {name} GARCH(1,1) ---")
    model = arch_model(df[name], mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal', rescale=True)
    # print iteration info every iter, allow up to 1000 its
    res = model.fit(cov_type='classic', update_freq=10, disp='off', options={'maxiter':1000}) # cov_type='robust'
    garch_res[name] = res
    ω, α, β = res.params[['omega','alpha[1]','beta[1]']]
    print(res.summary())
    
    #print("\nAR - GARCH Model Results:")
    #print(f" AR(1): const: {res.params['Const']:.6f}, φ: {res.params[f'{name}[1]']:.6f}")
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
# Wald test H0: α+β = 1 vs H1: α+β < 1 (one‐sided) MUST be corrected, 
# I'm sure this is NOT the right way to do it !!!

# TA: The constraint enforced by the arch package might affect the validity 
#     of the Wald test, since the boundary is excluded during the estimation.
# ----------------------------------------------------------------------------

# Volatility forecast (52 weeks)
horizon = 52
# 1) Unconditional variances
uc_s = garch_res['r_s'].params['omega'] / (1 - garch_res['r_s'].params['alpha[1]'] - garch_res['r_s'].params['beta[1]'])
uc_b = garch_res['r_b'].params['omega'] / (1 - garch_res['r_b'].params['alpha[1]'] - garch_res['r_b'].params['beta[1]'])
# 2) Conditional variance forecasts for next 52 weeks
vf_s = garch_res['r_s'].forecast(horizon=horizon, reindex=False).variance.iloc[0].values
vf_b = garch_res['r_b'].forecast(horizon=horizon, reindex=False).variance.iloc[0].values
# 3) Plotting
weeks = np.arange(1, horizon+1)
plt.figure(figsize=(12, 4))
# Stock volatility forecast
plt.subplot(1, 2, 1)
plt.plot(weeks, np.sqrt(vf_s) * 100, label='Conditional Vol')
plt.plot(weeks, np.sqrt(uc_s) * 100 * np.ones(horizon), '--', label='Unconditional Vol')
plt.title('Stock Weekly Volatility Forecast')
plt.xlabel('Weeks Ahead')
plt.ylabel('Volatility (%)')
plt.legend()
# Bond volatility forecast
plt.subplot(1, 2, 2)
plt.plot(weeks, np.sqrt(vf_b) * 100, label='Conditional Vol')
plt.plot(weeks, np.sqrt(uc_b) * 100 * np.ones(horizon), '--', label='Unconditional Vol')
plt.title('Bond Weekly Volatility Forecast')
plt.xlabel('Weeks Ahead')
plt.ylabel('Volatility (%)')
plt.legend()
plt.tight_layout()
plt.show()


###############################################################################
# PART 3: DYNAMIC ALLOCATION
###############################################################################

# -----------------------------------------------------------------------------
# Q3.1: Compute & plot static vs dynamic optimal weights (λ=2,10)
# -----------------------------------------------------------------------------

# residual cross‐correlation
eps_s = ar1_res['r_s'].resid
eps_b = ar1_res['r_b'].resid
rho_sb = eps_s.corr(eps_b)

# one‐step GARCH variance forecasts
fcast_s = garch_res['r_s'].forecast(horizon=1, reindex=False).variance['h.1']
fcast_b = garch_res['r_b'].forecast(horizon=1, reindex=False).variance['h.1']

# assemble forecasts into DataFrame aligned with df
fcast = pd.DataFrame({'sigma2_s': fcast_s, 'sigma2_b': fcast_b})

# prepare dynamic weights storage
dynamic_weights = {lam: pd.DataFrame(index=fcast.index, columns=['alpha_s','alpha_b']) for lam in lambdas}

# AR(1) params
a_s, rho_s = ar1_res['r_s'].params
a_b, rho_b = ar1_res['r_b'].params

for t in fcast.index:
    # expected returns next period
    mu_t1 = np.array([
        a_s + rho_s * df.at[t,'r_s'],
        a_b + rho_b * df.at[t,'r_b']
    ])
    rf_t = df.at[t,'r_f']
    diff = mu_t1 - rf_t
    
    # expected covariance next period
    σ2_s = fcast.at[t,'sigma2_s']
    σ2_b = fcast.at[t,'sigma2_b']
    σ_sb = rho_sb * np.sqrt(σ2_s * σ2_b)
    Σ_t1 = np.array([[σ2_s, σ_sb],
                     [σ_sb, σ2_b]])
    Σ_inv = np.linalg.inv(Σ_t1)
    
    # compute weights for each λ
    for lam in lambdas:
        dynamic_weights[lam].loc[t] = Σ_inv.dot(diff) / lam

# PLOT
fig, axes = plt.subplots(len(lambdas), 2, figsize=(12, 6), sharex=True)
for i, lam in enumerate(lambdas):
    # stock weight
    axes[i,0].hlines(static_weights.at[lam,'alpha_s'], df.index[0], df.index[-1],
                     colors='black', linestyles='--', label='static')
    dynamic_weights[lam]['alpha_s'].astype(float).plot(ax=axes[i,0], label='dynamic')
    axes[i,0].set_title(f'λ={lam}: α_s weights')
    axes[i,0].legend()
    # bond weight
    axes[i,1].hlines(static_weights.at[lam,'alpha_b'], df.index[0], df.index[-1],
                     colors='black', linestyles='--', label='static')
    dynamic_weights[lam]['alpha_b'].astype(float).plot(ax=axes[i,1], label='dynamic')
    axes[i,1].set_title(f'λ={lam}: α_b weights')
    axes[i,1].legend()

plt.tight_layout()
plt.show()





import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, chisquare
from statsmodels.stats.diagnostic import acorr_ljungbox

print("\n=== Q2.3: AR(1)-GARCH(1,1) with adequacy tests ===\n")
for name in ['r_s','r_b']:
    print(f"--- {name} GARCH(1,1) ---")
    model = arch_model(df[name],
                       mean='AR', lags=1,
                       vol='GARCH', p=1, q=1,
                       dist='normal', rescale=True)
    res = model.fit(cov_type='classic',
                    update_freq=10, disp='on',
                    options={'maxiter':1000})
    
    y = df[name].dropna()                  # your input series
    z = pd.Series(res.std_resid,           # numpy array
              index=y.index)           # re-use the same index
    u = pd.Series(norm.cdf(z),             # turn into PITs
              index=z.index)

    
    # parameters
    ω, α, β = res.params[['omega','alpha[1]','beta[1]']]
    print(res.summary())
    print(f"ω={ω:.6f}, α={α:.6f}, β={β:.6f}, α+β={(α+β):.4f}\n")

    # --- 1) Build PITs of standardized residuals ---
    # std_resid is ε_t / σ̂_t
    z = pd.Series(res.std_resid, index=res.model._y.index).dropna()
    u = pd.Series(norm.cdf(z), index=z.index)

    T = len(u)

    # --- 2) LM (Ljung‐Box) test for serial corr. in u_t ---
    lb = acorr_ljungbox(u, lags=[4], return_df=True).iloc[0]
    LM_stat = lb['lb_stat']
    pval_lm = lb['lb_pvalue']
    print("LB(4) stat, p‐value:", lb['lb_stat'], lb['lb_pvalue'])
    
    counts, _ = np.histogram(u, bins=10, range=(0,1))
    exp = len(u)/10
    chi2_stat, pval = chisquare(counts, f_exp=[exp]*10)
    print("Pearson χ², p‐value:", chi2_stat, pval)
    
    # --- 3) Pearson χ² test for uniformity of u_t ---
    N = 10
    counts, _ = np.histogram(u, bins=np.linspace(0,1,N+1))
    expected = T / N
    χ2_stat, pval_uni = chisquare(counts, f_exp=np.repeat(expected, N))

    # --- 4) ±2σ bin‐count bands under Binomial(T,1/N) ---
    se = np.sqrt(T * (1/N) * (1 - 1/N))
    lower, upper = expected - 2*se, expected + 2*se

    # --- Report ---
    print("Adequacy of density forecasts (u_t):")
    print(f"  Ljung–Box(4) stat = {LM_stat:.2f},  p = {pval_lm:.3f}")
    print(f"  Pearson χ²( N–1 ) = {χ2_stat:.2f},  p = {pval_uni:.3f}")
    print(f"  Bin‐count 95% bands: {lower:.1f} to {upper:.1f}")
    print(f"  Observed counts: {counts}\n")
