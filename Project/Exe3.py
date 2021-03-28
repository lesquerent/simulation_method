import numpy as np
from scipy.stats import norm


# -------- Not operational

def function_A(K, S0, r, sigma, T):
    return (np.log(K / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * T)


def g(x, K, S, r, sigma, T):
    A = function_A(K, S0, r, sigma, T)
    return A * np.exp(-A * (x - A))


def f(x, K, S, r, sigma, T):
    A = function_A(K, S0, r, sigma, T)
    return np.exp(-0.5 * x ** 2) / (1 - np.sqrt(2 * np.pi) * norm.cdf(A))


if __name__ == "__main__":
    S0 = 100
    K = 1000000
    T = 1.0
    sigma = 0.2
    r = 0.01

    samples = []
    A = function_A(K, S0, r, sigma, T)
    for i in range(100):

        u = np.random.uniform(0, 1)
        y = np.exp(A ** 2) * np.random.exponential(A)

        print(y)

        if u < A * f(y, K, S0, r, sigma, T) / g(y, K, S0, r, sigma, T):
            samples.append(y)

#--------