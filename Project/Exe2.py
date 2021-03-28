import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def pricing_call_option_with_BS(_spot_price, _strike_price, _risk_free_rate, _volatility, _maturity, _dividend_yield=0):
    """
    Return the price off an European call option with the Black-Scholes method

    Parameters
    ----------
    _spot_price : DOUBLE
        Value of the underlying's price today.
    _strike_price : DOUBLE
        Strike value.
    _maturity : DOUBLE
        Maturity of the option.
    _volatility : DOUBLE
        Expected annualized volatility of the underlying during the period..
    _risk_free_rate : DOUBLE
        Risk free rate.
    _dividend_yield : DOUBLE
        Expected dividend yield. Default : 0


    Returns
    -------
    price : DOUBLE
        Value of the option.

    """
    d1 = (np.log(_spot_price / _strike_price) + _maturity * (
            _risk_free_rate - _dividend_yield + _volatility ** 2 / 2)) / (
                 _volatility * np.sqrt(_maturity))
    d2 = d1 - _volatility * np.sqrt(_maturity)

    bs_call_price = _spot_price * np.exp(-_dividend_yield * _maturity) * stats.norm.cdf(d1) - _strike_price * np.exp(
        -_risk_free_rate * _maturity) * stats.norm.cdf(d2)

    return bs_call_price


def pricing_digital_call_option_with_BS(_spot_price, _strike_price, _maturity, _volatility, _risk_free_rate,
                                        _dividend_yield=0):
    """
    Return the price off an European digital call option with the Black-Scholes method

    Parameters
    ----------
    _spot_price : DOUBLE
        Value of the underlying's price today.
    _strike_price : DOUBLE
        Strike value.
    _maturity : DOUBLE
        Maturity of the option.
    _volatility : DOUBLE
        Expected annualized volatility of the underlying during the period..
    _risk_free_rate : DOUBLE
        Risk free rate.
    _dividend_yield : DOUBLE
        Expected dividend yield. Default : 0


    Returns
    -------
    price : DOUBLE
        Value of the option.

    """

    x = (np.log(_strike_price) - np.log(_spot_price) - (_risk_free_rate - (1 / 2) * _volatility ** 2) * _maturity) \
        / (_volatility * np.sqrt(_maturity))

    price = np.exp(-_risk_free_rate * _maturity) * stats.norm.cdf(-x)
    return price


def importance_sampling_pricing(S0, K, r, sigma, T, n):
    # Risk neutral measure
    u = np.random.normal(size=n)
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
    spot_price = 100
    strike_price_ATM = 100
    strike_price_OTM = 1_000_000
    maturity = 1.0
    volatility = 0.2
    risk_free_rate = 0.01
    nb_sequence = 100_000

    # ATM Pricing
    call_price_bs_ATM = pricing_call_option_with_BS(spot_price, strike_price_ATM, risk_free_rate, volatility, maturity)
    digital_call_price_bs_ATM = pricing_digital_call_option_with_BS(spot_price, strike_price_ATM, risk_free_rate,
                                                                    volatility, maturity)

    atm_ratio = call_price_bs_ATM / digital_call_price_bs_ATM

    print('### ATM PRICING WITH BLACK-SCHOLES ###')
    print('The Call price ATM with Black-Scholes is : {}'.format(call_price_bs_ATM))
    print('The Digital Call price ATM with Black-Scholes is : {}'.format(digital_call_price_bs_ATM))
    print('Ratio value : {}'.format(atm_ratio))

    # OTM Pricing
    call_price_bs_OTM = pricing_call_option_with_BS(spot_price, strike_price_OTM, risk_free_rate, volatility, maturity)
    digital_call_price_bs_OTM = pricing_digital_call_option_with_BS(spot_price, strike_price_OTM, risk_free_rate,
                                                                    volatility, maturity)
    otm_ratio = call_price_bs_OTM / digital_call_price_bs_OTM

    print('\n### OTM PRICING WITH BLACK-SCHOLES ###')
    print('The Call price OTM with Black-Scholes is : {}'.format(call_price_bs_OTM))
    print('The Digital Call price OTM with Black-Scholes is : {}'.format(digital_call_price_bs_OTM))
    print('Ratio value with Black-Scholes : {}'.format(otm_ratio))

    # OTM Pricing with importance sampling
    is_pricing = importance_sampling_pricing(spot_price, strike_price_OTM, risk_free_rate, volatility, maturity,
                                             nb_sequence)

    is_european_call = is_pricing[0]
    is_digital_call = is_pricing[1]
    is_ratio = is_european_call / is_digital_call
    is_S_T = is_pricing[2]
    is_S_T_ = is_pricing[3]

    # Risk neutral measure
    plt.title('Price distribution under risk neutral measure')
    plt.hist(is_S_T, 100)
    plt.show()

    # Drifting the measure with mu
    plt.title('Price distribution under mu drift measure')
    plt.hist(is_S_T_, 100, color='orange')
    plt.show()

    print('\n### OTM PRICING WITH IMPORTANCE SAMPLING ###')
    print('The Digital Call price OTM with Importance Sampling is : {}'.format(is_digital_call))
    print('The Call price OTM with Importance Sampling is : {}'.format(is_european_call))

    print('Ratio value with Importance Sampling : {}'.format(is_ratio))

