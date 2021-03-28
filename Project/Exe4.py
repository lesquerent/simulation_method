import numpy as np
import scipy.stats
from scipy import stats

from Project.Exe2 import importance_sampling_pricing


def b_ary(nb_expansion, base):
    """
    Parameters
    ----------
    nb_expansion : array_like
        The array of expansion series
    base : int
        Base of the expansion
    Returns
    -------
    nb_expansion : array_like
         The nb expansion
    """

    list_index_to_increment = np.where(nb_expansion < base - 1)

    if len(list_index_to_increment[0] != 0):
        index_to_increment = list_index_to_increment[0][0]
        nb_expansion[0:index_to_increment] = 0
        nb_expansion[index_to_increment] += 1

    else:
        nb_expansion = nb_expansion * 0
        nb_expansion = np.append(nb_expansion, 1)

    return nb_expansion


def generate_k_b_ary_expansion(k, base=10):
    """
        Generate an array with k expansion
    :param k: int
        The number of value in the expansion
    :param base: int
        The base of calculation. Default : 10
    :return: array_like
        The array with the k expansion
    """
    nb_exp = np.zeros(int(np.log(k) / np.log(base)) + 1)
    k_b_ary_expansion = np.array([b_ary(nb_exp, base).copy() for i in range(k)])
    # k_b_ary_expansion = np.array([b_ary(nb_exp, base) for i in range(k)])
    return k_b_ary_expansion


def van_der_corput_sequence(k, base):
    """
        Calculate the van der corput sequence of k number for the base selected
    :param k: int
        The number of value in the expansion
    :param base: int
        The base of calculation. Default : 10
    :return: array_like
        The list of the Van Der Corput Sequences
    """
    k_b_ary_expansion = generate_k_b_ary_expansion(k, base)
    n = len(k_b_ary_expansion[-1])

    bj = np.linspace(1, n, n)
    bj = 1 / (base ** bj)

    return k_b_ary_expansion.dot(bj)


def importance_sampling_qmc_pricing(S0, K, r, sigma, T, n):
    # Risk neutral measure
    array_of_rand_VDC_seq = van_der_corput_sequence(n, 2)
    u = scipy.stats.norm.ppf(array_of_rand_VDC_seq)

    is_S_T = S0 * np.exp((r - sigma * sigma / 2) * T + sigma * u)

    # Drifting the measure with mu
    u = np.random.normal(size=n)
    mu = 9.2104
    is_S_T_ = S0 * np.exp((mu - sigma * sigma / 2) * T + sigma * u)

    # Radon-nikodym derivative
    is_radon_nikodym_der = np.exp(
        -((np.log(is_S_T_ / S0) - (r - sigma * sigma / 2) * T) ** 2) / (2 * sigma * sigma * T) + (
                (np.log(is_S_T_ / S0) - (mu - sigma * sigma / 2) * T) ** 2) / (2 * sigma * sigma * T) + 500)

    # Two types of Calls
    is_european_call = np.mean(np.maximum(is_S_T_ - K, 0) * is_radon_nikodym_der)
    is_digital_call = np.mean(np.maximum(is_S_T_ - K, 0) * is_radon_nikodym_der * 1 / (is_S_T_ - K))

    return is_european_call, is_digital_call, is_S_T, is_S_T_


if __name__ == '__main__':
    k = 10
    base2 = 2
    base10 = 10

    test_bary = generate_k_b_ary_expansion(k, 2)
    print('The b-ary expansion with k = {}, base = {} is : \n{}'.format(k, base2, test_bary))

    test_b10 = van_der_corput_sequence(k, base10)
    test_b2 = van_der_corput_sequence(k, base2)

    print("\nThe Van Der Corput Sequences with base 2 is :\n{}".format(test_b2))
    print()
    print("The Van Der Corput Sequences with base 10 is :\n{}".format(test_b10))

    spot_price = 100
    strike_price_ATM = 100
    strike_price_OTM = 1_000_000
    maturity = 1.0
    volatility = 0.2
    risk_free_rate = 0.01
    nb_sequence = 100_000

    # OTM Pricing with importance sampling
    is_pricing = importance_sampling_pricing(spot_price, strike_price_OTM, risk_free_rate, volatility, maturity,
                                             nb_sequence)

    is_european_call = is_pricing[0]
    is_digital_call = is_pricing[1]
    is_ratio = is_european_call / is_digital_call
    is_S_T = is_pricing[2]
    is_S_T_ = is_pricing[3]

    print('\n### OTM PRICING WITH IMPORTANCE SAMPLING ###')
    print('The Digital Call price OTM with Importance Sampling is : {}'.format(is_digital_call))
    print('The Call price OTM with Importance Sampling is : {}'.format(is_european_call))
    print('Ratio value with Importance Sampling : {}'.format(is_ratio))

    # OTM Pricing with importance sampling
    is_qmc_pricing = importance_sampling_qmc_pricing(spot_price, strike_price_OTM, risk_free_rate, volatility, maturity,
                                                     nb_sequence)

    is_qmc_european_call = is_qmc_pricing[0]
    is_qmc_digital_call = is_qmc_pricing[1]
    is_qmc_ratio = is_qmc_european_call / is_qmc_digital_call
    # is_qmc_S_T = is_pricing[2]
    is_qmc_S_T_ = is_pricing[3]
    is_qmc_standard_error = stats.sem(is_qmc_S_T_)

    print('\n### OTM PRICING WITH QMC IN IMPORTANCE SAMPLING ###')
    print('The Digital Call price OTM with Importance Sampling is : {}'.format(is_qmc_digital_call))
    print('The Call price OTM with Importance Sampling is : {}'.format(is_qmc_european_call))

    print('Ratio value with Importance Sampling : {}'.format(is_qmc_ratio))
    print('Standard error of the price with the new dynamic with QMC : {}'.format(is_qmc_standard_error))
