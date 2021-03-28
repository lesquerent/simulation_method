import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import scipy.stats as sc


def clayton_copula(n, tau):
    theta = 2 * tau / (1 - tau)

    U = np.random.uniform(size=n)
    W = np.random.uniform(size=n)
    V = U * (W ** (-theta / (1 + theta)) - 1 + U ** theta) ** (-1 / theta)
    return U, V


def resample(A1, A2, n):
    res1 = np.empty(n)
    res2 = np.empty(n)
    for i in range(n):
        u = np.random.randint(0, len(data) - 1)
        res1[i] = A1[u]
        res2[i] = A2[u]
    df = pd.DataFrame({'Asset1': res1, 'Asset2': res2})
    return df


def bootstrapKendallTauEstimator(df):
    corr = df.corr()
    tau, p_value = sc.kendalltau(df['Asset1'], df['Asset2'])
    return tau


def bootstrapSpearmanTauEstimator(df):
    tau, p_value = sc.spearmanr(df['Asset1'], df['Asset2'])
    return tau


def bootstrap(data):
    M = 1000
    n = 500
    Kendall_tau = np.empty(M)
    Spearman_tau = np.empty(M)
    for i in range(M):
        res = resample(data["Asset1"].to_numpy(), data["Asset2"].to_numpy(), n)
        K_tau = bootstrapKendallTauEstimator(res)
        Kendall_tau[i] = K_tau
        S_tau = bootstrapSpearmanTauEstimator(res)
        Spearman_tau[i] = S_tau
    return Kendall_tau, Spearman_tau


if __name__ == "__main__":
    # Question 1
    # ----------

    header_list = ["Asset1", "Asset2"]
    data = pd.read_csv('data_simulation_methods.csv', names=header_list)

    # Price visualization
    data.plot(figsize=(15, 10), xlabel='Time', ylabel='Price', title='Price of different asset in function of time')
    plt.show()

    # DataFrame of log returns
    log_ret = data.apply(lambda x: np.log(x) - np.log(x.shift(1))).drop(0)

    # Drift
    drift = log_ret.mean()

    # Volatility
    vol = log_ret.std()

    # Log return correlation
    cor = log_ret.corr()

    # Question 2
    # ----------

    x_axis = np.arange(-0.05, 0.05, 0.000001)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Histogram of log return distribution')
    ax1, ax2 = axes.flatten()

    ax1.set_title('Asset 1')
    ax1.hist(log_ret['Asset1'], bins=80, color="DarkSeaGreen")
    ax1.plot(x_axis, norm.pdf(x_axis, drift['Asset1'], vol['Asset1']), c='OrangeRed')

    ax2.set_title('Asset 2')
    ax2.hist(log_ret['Asset2'], bins=70, color="LightSeaGreen")
    ax2.plot(x_axis, norm.pdf(x_axis, drift['Asset2'], vol['Asset2']), c='OrangeRed')

    plt.show()

    # Question 3
    # ----------

    h = sns.jointplot(log_ret['Asset1'], log_ret['Asset2'], kind='kde', stat_func=None)
    h.set_axis_labels('Asset 1', 'Asset 2', fontsize=16)
    sns.scatterplot(x=log_ret['Asset1'], y=log_ret['Asset2'])

    n = 10000
    mean = [0, 0]
    cov = [(1, .7), (.7, 1)]
    rng = np.random.RandomState(0)
    x, y = rng.multivariate_normal(mean, cov, n).T

    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, s=5, color=".15")
    plt.show()

    # Tau Value
    Kendall_tau, p_value = sc.kendalltau(log_ret['Asset1'], log_ret['Asset2'])
    Spearman_tau, p_value2 = sc.spearmanr(log_ret['Asset1'], log_ret['Asset2'])

    # Question 4
    # ----------

    # Tau Value
    tau, p_value = sc.kendalltau(log_ret['Asset1'], log_ret['Asset2'])
    print(tau, p_value)

    uc, vc = clayton_copula(1000, tau)
    plt.scatter(uc, vc, marker='.', color='red')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Clayton copula')
    plt.show()

    # Question 5
    # ----------
    a1 = norm.ppf(uc)
    a2 = norm.ppf(vc)

    a1_price = [(10 ** (a1[0] * 0.01)) * data['Asset1'][999]]
    a2_price = [(10 ** (a1[0] * 0.01)) * data['Asset2'][999]]
    for i in range(1, len(a1)):
        a1_price.append((10 ** (a1[i] * 0.01)) * a1_price[i - 1])
        a2_price.append((10 ** (a2[i] * 0.01)) * a2_price[i - 1])

    t = np.linspace(0, 2000, 2000)
    t2 = np.linspace(0, 1000, 1000)

    plt.plot(t, np.concatenate((data['Asset1'], a1_price)), c='PaleTurquoise')
    plt.plot(t, np.concatenate((data['Asset2'], a2_price)), c="PeachPuff")
    plt.plot(t2, data['Asset1'])
    plt.plot(t2, data['Asset2'])
    plt.title("Evolution of price according to clayton copula")
    plt.show()

    # SUMMARY OF MODELISATION

    # Drift
    drift_prev = [a1.mean(), a2.mean()]

    # Volatility
    vol_prev = [a1.std(), a2.std()]

    k = pd.DataFrame()
    k['Prevision_Asset1'] = a1
    k['Prevision_Asset2'] = a2

    # Log return correlation
    cor_prev = k.corr()

    print(drift_prev, vol_prev, cor_prev)

    # Question 6
    # ----------

    simulated_prices = pd.DataFrame({'Asset1': a1_price, 'Asset2': a2_price})
    simulated_log_ret = simulated_prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).drop(0)

    # Delta is our statistic measure for all resample set
    Kendall_tau_bootstrap, Spearman_tau_bootstrap = bootstrap(simulated_log_ret)
    print("kendall Tau for simulated prices using bootstrap : " + str(np.mean(Kendall_tau_bootstrap)))
    print("Kendall Tau with the original sample : " + str(Kendall_tau))
    print("Their difference : " + str(np.mean(Kendall_tau_bootstrap) - Kendall_tau))
    print("\n")
    print("Spearman Tau for simulated prices using bootstrap : " + str(np.mean(Spearman_tau_bootstrap)))
    print("Spearman Tau with the original sample : " + str(Spearman_tau))
    print("Their difference : " + str(np.mean(Spearman_tau_bootstrap) - Spearman_tau))
