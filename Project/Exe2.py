from import_packages import *


def pricing_call_option_with_BS(spot_price, strike_price, maturity, volatility, risk_free_rate, dividend_yield=0):
    """
    Return the price off an European call option with the Black-Scholes method

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
        Expected dividend yield. Default : 0


    Returns
    -------
    price : DOUBLE
        Value of the option.

    """

    d1 = (np.log(spot_price / strike_price) + maturity * (risk_free_rate - dividend_yield + volatility ** 2 / 2)) / (
            volatility * np.sqrt(maturity))

    d2 = d1 - volatility * np.sqrt(maturity)

    price = spot_price * np.exp(-dividend_yield * maturity) * stats.norm.cdf(d1) - strike_price * np.exp(
        -risk_free_rate * maturity) * stats.norm.cdf(d2)

    return price


def pricing_digital_call_option_with_BS(spot_price, strike_price, maturity, volatility, risk_free_rate,
                                        dividend_yield=0):
    """
    Return the price off an European digital call option with the Black-Scholes method

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
        Expected dividend yield. Default : 0


    Returns
    -------
    price : DOUBLE
        Value of the option.

    """

    x = (np.log(strike_price) - np.log(spot_price) - (risk_free_rate - (1 / 2) * volatility ** 2) * maturity) \
        / (volatility * np.sqrt(maturity))

    price = np.exp(-_strike_price * maturity) * stats.norm.cdf(-x)
    return price


# Variables definition
_spot_price = 100
_strike_price = 100000
_maturity = 1.0
_volatility = 0.2
_risk_free_rate = 0.01

_nb_sequences = 100000
_nb_price = 1

## Results

call_price = pricing_call_option_with_BS(_spot_price, _strike_price, _maturity, _volatility, _risk_free_rate)
print('The price for this call far from the money is {}'.format(call_price))
# Result : The price for this call far from the money is 1.0709944150897396e-259

digital_call_price = pricing_digital_call_option_with_BS(_spot_price, _strike_price, _maturity, _volatility,
                                                         _risk_free_rate)
print('The price for this digital call far from the money is {}'.format(digital_call_price))
# Result : The price for this digital call far from the money is 0.0
