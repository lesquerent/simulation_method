# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:50:57 2021

TD1 Simulation methods

@author: thiba
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from math import exp, sqrt


# Exercise 1
def box_Muller():
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2


def modified_Box_Muller():
    z = 1
    u1, u2 = 0, 0

    # Rejection
    while (z >= 1):
        u1 = np.random.uniform(-1, 1)
        u2 = np.random.uniform(-1, 1)

        z = (u1 ** 2) + (u2 ** 2)

    # Acceptance
    r = -2 * np.log(z)

    x = np.sqrt(r / z) * u1
    y = np.sqrt(r / z) * u2

    return x, y


# Exercise 2
def acc_rej_method(pdf, n=500, xmin=0, xmax=1):
    """
    :param pdf: fonction de répartition utilisée
    :param n: nombre de subdivision
    :param xmin: x minimum
    :param xmax: x maximum
    :return: retourne un graphique des points dans la pdf ou non
    """
    M = np.sqrt(2 / np.pi) * np.exp(1 / 2)
    xIn, yIn = [], []
    xOut, yOut = [], []

    for i in range(n):
        y = np.random.exponential(1)
        u = np.random.uniform(0, M * pdf(y))
        if u < pdf(y):
            xIn.append(y)
            yIn.append(u)
        else:
            xOut.append(y)
            yOut.append(u)

    plt.scatter(xIn, yIn, c='red')
    plt.scatter(xOut, yOut, c='green')

    x = np.linspace(xmin, 10, 1000)
    res = []
    for e in x:
        res.append(pdf(e))
    plt.plot(x, res)
    plt.show()
    return 1


# Exercise 3
def pricing_option_with_MC_method(spot_price, strike_price, volatility, maturity, risk_free_rate, nb_sequences,
                                  nb_price, option_type='call'):
    """
    Price a Call or Put option, returns the price of the option, the array of all prices during the simulation
    and the mean of all pay off

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
        Number of prices per sequences.
    option_type : STR, optional
        Option type, call or put. The default is 'CALL'.


    Returns
    -------
    option_price : DOUBLE
        Value of the option.
    array_of_all_prices : ARRAY_LIKE
        Array with all prices.
    mean_pay_off : DOUBLE
        The mean of all pay off at maturity.
    array_of_payoff : ARRAY_LIKE
        The array with the payoff at maturity
    """
    if option_type not in ('call', 'put'):
        raise ValueError('option parameter must be one of "call" (default) or "put".')

    delta_t = maturity / nb_price
    array_of_normal = np.random.normal(0, 1, (nb_sequences, nb_price))
    array_of_variation = ((risk_free_rate - ((volatility ** 2) / 2)) * delta_t) + (volatility * sqrt(delta_t)
                                                                                   * array_of_normal)

    cumulated_variation = np.cumsum(array_of_variation, axis=1)

    array_of_all_prices = spot_price * np.exp(cumulated_variation)
    # add S0 in all prices array
    array_of_all_prices = np.c_[spot_price * np.ones(array_of_all_prices.shape[0]), array_of_all_prices]

    if option_type == 'call':
        array_of_payoff = np.maximum(array_of_all_prices[:, -1] - strike_price, 0)
        mean_payoff = np.mean(array_of_payoff)
    else:
        array_of_payoff = np.maximum(strike_price - array_of_all_prices[:, -1], 0)
        mean_payoff = np.mean(array_of_payoff)

    option_price = mean_payoff * np.exp(-risk_free_rate * maturity)

    return option_price, array_of_all_prices, mean_payoff, array_of_payoff


def calcul_error(array_of_payoff, percent=99):
    """
    Calculate the error of the pay off estimation.

    Parameters
    ----------
    array_of_payoff : ARRAY_LIKE
        Array of pay off.
    percent : FLOAT, optional
        Percent of confidence, three choices, 99,95,90. The default is 99.

    Returns
    -------
    error : DOUBLE
        Value of the error.

    """
    if percent not in (99, 95, 90):
        raise ValueError('Percent optional parameter must be one of int(99) (default) or int(95), int(90).')

    nb_sequences = np.shape(array_of_payoff)[0]

    mean_payoff = np.mean(array_of_payoff)
    mean_squared_payoff = np.mean(array_of_payoff ** 2)

    variance = mean_squared_payoff - (mean_payoff ** 2)

    if percent == 99:
        q = 2.5758
        error = q * sqrt(variance / nb_sequences)

    elif percent == 95:
        q = 1.9600
        error = q * sqrt(variance / nb_sequences)

    else:  # percent == 90
        q = 1.6449
        error = q * sqrt(variance / nb_sequences)

    return mean_payoff - error


def get_confidence_interval(array_of_payoff, option_price, percent=99, rounded=True):
    """
    Return a string corresponding to the interval of confidence

    Parameters
    ----------
    array_of_payoff : ARRAY_LIKE
        Array of pay off.
    option_price : DOUBLE
        Estimated price of the option.
    percent : FLOAT, optional
        Percent of confidence, three choices, 99,95,90. The default is 99.
    rounded : BOOL, optional
        If we want to round the price et the error, The default is True

    Returns
    -------
    interval : STR
        Interval of confidence.

    """

    error = calcul_error(array_of_payoff, percent)
    error_value = abs(option_price - error)

    if rounded:
        option_price = round(option_price, 2)
        error_value = round(error_value, 2)

    interval = "[{price} +- {error}]".format(price=option_price, error=error_value)

    return interval


def display_MC(all_prices, mean_payoff, option_price, array_of_payoff, maturity, option_type='call', percent=99):
    """
    Display the price evolution.

    Parameters
    ----------
    all_prices : ARRAY_LIKE
        Array with all prices.
    mean_payoff : DOUBLE
        The mean of all pay off at maturity.
    option_price : DOUBLE
        Value of the option.
    maturity : DOUBLE
        Maturity of the option.
    option_type : STR, optional
        Option type, call or put. The default is 'call'.
    percent : FLOAT, optional
        Percent of confidence, three choices, 99,95,90. The default is 99.

    Returns
    -------
    None.

    """
    nb_of_prices = len(all_prices[0])
    indexes = np.linspace(0, maturity, nb_of_prices)
    confidence_interval = get_confidence_interval(array_of_payoff, option_price, percent)

    plt.axis([indexes.min(), indexes.max(), all_prices.min() - 1, all_prices.max() + 1])
    plt.xlabel("Temps.")
    plt.ylabel("Value of the {type} option".format(type=option_type))
    plt.title("Price estimation with Monte Carlo method.")

    plt.plot([], color='w', label='Avg PayOff :' + str(round(mean_payoff, 2)) + "€")

    plt.plot([], color='w', label='Pricing :' + str(round(option_price, 2)) + "€")

    plt.plot([], color='w', label='Confidence interval at ' + str(percent) + "%" + confidence_interval)

    plt.plot(indexes.T, all_prices.T)

    plt.legend()
    plt.grid()
    plt.show()


def pricing_option_with_BS(spot_price, strike_price, maturity, volatility, risk_free_rate, dividend_yield,
                           option_type='call'):
    """
    Return the price off an European option with the Black-Scholes method

    Parameters
    ----------
    spot_price : DOUBLE
        Value of the underlying's price today.
    strike_price : DOUBLE
        Strike value.
    maturity : DOUBLE
        Maturity of the option.
    volatility : DOUBLE
        Expected annualized volatility of the underlying during the period..
    risk_free_rate : DOUBLE
        Risk free rate.
    dividend_yield : DOUBLE
        Expected dividend yield.
    option_type : STR, optional
        Option type, call or put. The default is 'call'.

    Returns
    -------
    price : DOUBLE
        Value of the option.

    """

    if option_type not in ('call', 'put'):
        raise ValueError('option parameter must be one of "call" (default) or "put".')

    d1 = (np.log(spot_price / strike_price) + maturity * (risk_free_rate - dividend_yield + volatility ** 2 / 2)) / (
            volatility * np.sqrt(maturity))

    d2 = d1 - volatility * np.sqrt(maturity)

    if option_type == 'call':
        price = spot_price * np.exp(-dividend_yield * maturity) * stats.norm.cdf(d1) - strike_price * np.exp(
            -risk_free_rate * maturity) * stats.norm.cdf(d2)
    else:
        price = strike_price * np.exp(-risk_free_rate * maturity) * stats.norm.cdf(-d2) - spot_price * np.exp(
            -dividend_yield * maturity) * stats.norm.cdf(-d1)

    return price


if __name__ == '__main__':
    # Exercise 1 Box Muller
    # bx = box_Muller()
    # print(bx[0])
    # print(bx[1])

    # mbx = modified_Box_Muller()
    # print(mbx[0])
    # print(mbx[1])

    # Exercise 2 Acceptance Rejection

    # Exercise 3 MC and BS
    nb_sequences = 10000
    nb_price = 1
    maturity = 1
    risk_free_rate = 0.01
    volatility = 0.2
    spot_price = 100
    strike_price = 100
    type_option = 'call'

    # Monte carlo pricing
    call_option_priced_with_MC_method = pricing_option_with_MC_method(spot_price, strike_price, volatility, maturity,
                                                                      risk_free_rate,
                                                                      nb_sequences, nb_price, type_option)

    all_price = call_option_priced_with_MC_method[1]
    mean_payoff = call_option_priced_with_MC_method[2]
    option_price = call_option_priced_with_MC_method[0]
    array_of_payoff = call_option_priced_with_MC_method[3]
    interval_of_confidence = get_confidence_interval(array_of_payoff, option_price)

    # Black Scholes pricing
    dividend_yield = 0
    call_option_priced_with_BS_method = pricing_option_with_BS(spot_price, strike_price, maturity, volatility,
                                                               risk_free_rate, dividend_yield)

    # Display results
    print('Pricing with MC : \nPrice = {price} \nConfidence interval = {interval}'.format(price=option_price,
                                                                                          interval=interval_of_confidence))
    print('Pricing with BS :\n Price = {}'.format(call_option_priced_with_BS_method))

    # display_MC(call_option_priced_with_MC_method[1], call_option_priced_with_MC_method[2],
    #           call_option_priced_with_MC_method[0], array_of_payoff, maturity)

    plt.hist(all_price[:, -1],100)
    plt.show()
