import scipy

from import_packages import *


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

    # expansion = nb_expansion.copy()
    list_index_to_increment = np.where(nb_expansion < base - 1)

    if len(list_index_to_increment[0] != 0):
        index_to_increment = list_index_to_increment[0][0]
        nb_expansion[0:index_to_increment] = 0
        nb_expansion[index_to_increment] += 1

    else:
        nb_expansion = nb_expansion * 0
        nb_expansion = np.append(nb_expansion, 1)

    # nb_expansion = expansion
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


def price_evolution_with_QMC_method_VDC_seq(spot_price, strike_price, volatility, maturity, risk_free_rate,
                                            nb_sequences,
                                            nb_price=1):
    """
    Return the array of all prices during the simulation

    Parameters
    ----------
    spot_price : DOUBLE
        Value of the underlying price today.
    strike_price : DOUBLE
         Strike value.
    volatility : DOUBLE
        Expected annualized volatility of the underlying during the period.
    risk_free_rate : DOUBLE
        Risk free rate.
    maturity : DOUBLE
        Maturity of the option.
    nb_sequences : INT
        Number of prices sequences.
    nb_price : INT
        Number of prices per sequences. Default : 1


    Returns
    -------
    array_of_all_prices : ARRAY_LIKE
        Array with all prices.

    """

    delta_t = maturity / nb_price

    array_of_rand_VDC_seq = van_der_corput_sequence(nb_sequences, 2)

    array_of_normal = scipy.stats.norm.ppf(array_of_rand_VDC_seq)

    array_of_variation = ((risk_free_rate - ((volatility ** 2) / 2)) * delta_t) + (volatility * sqrt(delta_t)
                                                                                   * array_of_normal)

    # print(array_of_variation)
    cumulated_variation = array_of_variation

    cumulated_variation = np.cumsum(array_of_variation)#, axis=1)

    array_of_all_prices = spot_price * np.exp(cumulated_variation)
    # add S0 in all prices array
    array_of_all_prices = np.c_[spot_price * np.ones(array_of_all_prices.shape[0]), array_of_all_prices]

    return array_of_all_prices


if __name__ == '__main__':
    k_ = 10
    base10 = 10
    base2 = 2

    # test_bary = generate_k_b_ary_expansion(k_, base2)
    # print('Test bary {}'.format(test_bary))
    # print(test_bary[-4:])
    #
    # test_b10 = van_der_corput_sequence(k_, base2)
    # print('VDC {}'.format(test_b10))
    # test_b2 = -van_der_corput_sequence(k_, base2)
    #
    # print("The Van Der Corput Sequences with base 2 is :\n{}".format(test_b2))
    # print()
    # print("The Van Der Corput Sequences with base 10 is :\n{}".format(test_b10))

    # Variables definition
    _spot_price = 100
    _strike_price = 100
    _maturity = 1.0
    _volatility = 0.2
    _risk_free_rate = 0.01

    _nb_sequences = 1000
    _nb_price = 1

    price_evolution = price_evolution_with_QMC_method_VDC_seq(_spot_price, _strike_price, _volatility, _maturity,
                                                              _risk_free_rate,
                                                              _nb_sequences)

    call_price_MC = np.mean(np.maximum(price_evolution - _strike_price, 0)) * np.exp(-_risk_free_rate * _maturity)

    digital_call_payoff = price_evolution.copy()
    digital_call_payoff[digital_call_payoff < _strike_price] = 0
    digital_call_payoff[digital_call_payoff >= _strike_price] = 1
    digital_call_price_MC = np.mean(digital_call_payoff) * np.exp(-_risk_free_rate * _maturity)

    # Result
    print('\nResult obtained with Monte Carlo classic method :\n')
    plt.hist(price_evolution, 100)
    plt.title("Repartition of price at maturity with Monte Carlo classic method.")
    plt.show()
    print('The price for this call with MC far from the money is {}'.format(call_price_MC))
    # Result : The price for this call with MC far from the money is 0.0
    print('The price for this digital call with MC far from the money is {}'.format(digital_call_price_MC))
    # Result : The price for this digital call with MC far from the money is 0.0
