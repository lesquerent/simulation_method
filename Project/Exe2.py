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

    price = np.exp(-_risk_free_rate * maturity) * stats.norm.cdf(-x)
    return price


def price_evolution_with_MC_method(spot_price, strike_price, volatility, maturity, risk_free_rate, nb_sequences,
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
    array_of_normal = np.random.normal(0, 1, (nb_sequences, nb_price))
    array_of_variation = ((risk_free_rate - ((volatility ** 2) / 2)) * delta_t) + (volatility * sqrt(delta_t)
                                                                                   * array_of_normal)
    cumulated_variation = np.cumsum(array_of_variation, axis=1)

    array_of_all_prices = spot_price * np.exp(cumulated_variation)
    # add S0 in all prices array
    array_of_all_prices = np.c_[spot_price * np.ones(array_of_all_prices.shape[0]), array_of_all_prices]

    return array_of_all_prices


# Variables definition
_spot_price = 100
_strike_price = 100000
_maturity = 1.0
_volatility = 0.2
_risk_free_rate = 0.01

_nb_sequences = 100000
_nb_price = 1

##############
# Black-Scholes
print('\nResult obtained with Black-Scholes classic method :\n')
call_price = pricing_call_option_with_BS(_spot_price, _strike_price, _maturity, _volatility, _risk_free_rate)
print('The price for this call with BS far from the money is {}'.format(call_price))
# Result : The price for this call with BS far from the money is 1.0709944150897396e-259

digital_call_price = pricing_digital_call_option_with_BS(_spot_price, _strike_price, _maturity, _volatility,
                                                         _risk_free_rate)
print('The price for this digital call with BS far from the money is {}'.format(digital_call_price))
# Result : The price for this digital call with BS far from the money is 1.8446069543524016e-262


# Calcul the ratio :
print('The ratio with Black-Scholes is {} digital call options for one call option.'.format(
    call_price / digital_call_price))
# Result : The ratio with Black-Scholes is 580.6084665151556 digital call options for one call option.

##################
# Monte Carlo :
price_evolution = price_evolution_with_MC_method(_spot_price, _strike_price, _volatility, _maturity, _risk_free_rate,
                                                 _nb_sequences)[:, -1]

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


# Calcul the ratio :
print('The ratio with Monte Carlo is {} digital call options for one call option.'.format(
    call_price_MC / digital_call_price_MC))
# Result : The ratio with Monte Carlo is nan digital call options for one call option.

###############
# Monte Carlo with Importance of sampling:
print('\nResult obtained with Monte Carlo and importance of sampling :\n')
mu = 6.95  # (pour k = 100 000)
price_evolution_IS = price_evolution_with_MC_method(_spot_price, _strike_price, _volatility, _maturity, mu,
                                                    _nb_sequences)[:, -1]

digital_call_payoff_IS = price_evolution_IS.copy()
digital_call_payoff_IS[digital_call_payoff_IS < _strike_price] = 0
digital_call_payoff_IS[digital_call_payoff_IS >= _strike_price] = 1


# Plot
fig, ax = plt.subplots()
ax.hist(price_evolution_IS, bins=100, label=['Price generated with Importance of sampling'])
ax.legend(loc='best', shadow=True)
plt.title('"Repartition of price at maturity with Importance of sampling."')
plt.show()

radon_nikodym_der = np.exp(
    -(np.log(price_evolution_IS / _spot_price) - (_risk_free_rate - _volatility ** 2 / 2) * _maturity) ** 2
    / (2 * _volatility ** 2 * _maturity) + (
            np.log(price_evolution_IS / _spot_price) - (mu - _volatility ** 2 / 2) * _maturity) ** 2
    / (2 * _volatility ** 2 * _maturity)
)

#Price
call_price_MC_IS = np.mean(np.maximum(price_evolution_IS - _strike_price, 0)*radon_nikodym_der) * np.exp(-_risk_free_rate * _maturity)
digital_call_price_MC_IS = np.mean(digital_call_payoff_IS*radon_nikodym_der) * np.exp(-_risk_free_rate * _maturity)

print('The price for this call with MC and importance of sampling far from the money is {}'.format(call_price_MC_IS))
# Result : The price for this call with MC and importance of sampling far from the money is 1.0820880961709169e-259
print('The price for this digital call with MC and importance of sampling far from the money is {}'.format(digital_call_price_MC_IS))
# Result : The price for this digital call with MC and importance of sampling far from the money is 1.862152663556151e-262


# Calcul the ratio :
print('The ratio with Monte Carlo and importance sampling is {} digital call options for one call option.'.format(
    call_price_MC_IS / digital_call_price_MC_IS))
# Result : The ratio with Monte Carlo and importance sampling is 576.5802384547685 digital call options for one call option.
