
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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, norm, gaussian_kde
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey, het_white
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import t
from itertools import permutations
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS, add_constant

###############################################################################
# PART 0: DIRECTORY AND DATA SET UP
###############################################################################

# Set working directory to the script's location
import os
import sys
import pandas as pd
import numpy as np

###############################################################################
# PART 0: DIRECTORY AND DATA SET UP
###############################################################################

# Set working directory to the script's location
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())

# File paths
input_file = os.path.join("Data", "dati_EMF.xlsx")


###############################################################################
# PART 1: LOAD AND PREPARE DATA
###############################################################################

# 1. Load the data from the Excel file
smi  = pd.read_excel(input_file, sheet_name='SMI',  parse_dates=['DATE'])
bond = pd.read_excel(input_file, sheet_name='Bond', parse_dates=['DATE'])
rate = pd.read_excel(input_file, sheet_name='Rate', parse_dates=['DATE'])

# 2. Compute simple weekly returns for stocks and bonds
smi['r_s'] = smi['SWISS MARKET (SMI) - TOT RETURN IND'].pct_change()
bond['r_b'] = bond['SW BENCHMARK 10 YEAR DS GOVT. INDEX - TOT RETURN IND'].pct_change()

# 3. Compute weekly risk-free rate (convert % to decimal and divide by 52)
rate['r_f'] = rate['SWISS FRANC S/T DEPO (FT/LSEG DS) - MIDDLE RATE'] / 52

# 4. Merge into a single DataFrame
df = (
    smi[['DATE', 'r_s']]
    .merge(bond[['DATE', 'r_b']], on='DATE')
    .merge(rate[['DATE', 'r_f']], on='DATE')
    .dropna()
    .set_index('DATE')
)

print("\nFirst few rows of merged data:")
print(df.head())


###############################################################################
# PART 2: SAMPLE STATISTICS AND OPTIMAL WEIGHTS
###############################################################################

# 5. Sample means and covariance matrix
mu      = df[['r_s', 'r_b']].mean().values       # vector of sample means
Rf_bar  = df['r_f'].mean()                       # average risk-free rate
Sigma   = df[['r_s', 'r_b']].cov().values        # sample covariance matrix

# 6. Compute optimal weights for λ = 2 and λ = 10
results = []
for lam in [2, 10]:
    alpha = np.linalg.inv(Sigma).dot(mu - Rf_bar * np.ones(2)) / lam
    results.append({
        'lambda': lam,
        'alpha_s': alpha[0],
        'alpha_b': alpha[1],
        'alpha_cash': 1 - alpha.sum()
    })

# 7. Display results
results_df = pd.DataFrame(results).set_index('lambda')
print("\nOptimal weights for λ = 2 and 10:")
print(results_df)
