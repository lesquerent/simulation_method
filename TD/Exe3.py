import time

from import_packages import *


# Black-Scholes Call
def pricing_option_with_MC_method(spot_price, strike_price, volatility, maturity, risk_free_rate, nb_sequences,
                                  nb_price):
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

    delta_t = maturity / nb_price
    array_of_normal = np.random.normal(0, 1, (nb_sequences, nb_price))
    array_of_variation = ((risk_free_rate - ((volatility ** 2) / 2)) * delta_t) + (volatility * sqrt(delta_t)
                                                                                   * array_of_normal)

    cumulated_variation = np.cumsum(array_of_variation, axis=1)

    array_of_all_prices = spot_price * np.exp(cumulated_variation)
    # add S0 in all prices array
    array_of_all_prices = np.c_[spot_price * np.ones(array_of_all_prices.shape[0]), array_of_all_prices]

    array_of_payoff = np.maximum(array_of_all_prices[:, -1] - strike_price, 0)
    mean_payoff = np.mean(array_of_payoff)

    option_price = mean_payoff * np.exp(-risk_free_rate * maturity)

    return option_price, array_of_all_prices, mean_payoff, array_of_payoff


def error_calculation(array_of_payoff, percent=99):
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

    error = error_calculation(array_of_payoff, percent)
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
    array_of_payoff: ARRAY_LIKE
        The array with payoff
    maturity : DOUBLE
        Maturity of the option.
    option_type : STR, optional
        Option type, call or put. The default is 'call'.
    percent : FLOAT, optional
        Percent of confidence, three choices, 99,95,90. The default is 99.

    Returns
    -------
    None.
    :param

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


if __name__ =='__main__':

    # Monte carlo pricing

    # Constant declaration
    nb_sequences_ = 10000
    nb_price_ = 1
    maturity_ = 1
    risk_free_rate_ = 0.01
    volatility_ = 0.2
    spot_price_ = 100
    strike_price_ = 100

    call_option_priced_with_MC_method = pricing_option_with_MC_method(spot_price_, strike_price_, volatility_, maturity_,
                                                                      risk_free_rate_,
                                                                      nb_sequences_, nb_price_)

    all_price = call_option_priced_with_MC_method[1]
    mean_payoff = call_option_priced_with_MC_method[2]
    option_price = call_option_priced_with_MC_method[0]
    array_of_payoff = call_option_priced_with_MC_method[3]
    interval_of_confidence = get_confidence_interval(array_of_payoff, option_price)

    # Black Scholes pricing
    dividend_yield_ = 0
    call_option_priced_with_BS_method = pricing_option_with_BS(spot_price_, strike_price_, maturity_, volatility_,
                                                               risk_free_rate_, dividend_yield_)
    print('\nPricing with BS :\n Price = {}'.format(call_option_priced_with_BS_method))
    # Display results
    print('Pricing with MC : \nPrice = {price} \nConfidence interval = {interval}'.format(price=option_price,
                                                                                          interval=interval_of_confidence))
    print('\nPricing with BS :\n Price = {}'.format(call_option_priced_with_BS_method))

    # display_MC(call_option_priced_with_MC_method[1], call_option_priced_with_MC_method[2],
    #            call_option_priced_with_MC_method[0], array_of_payoff, maturity_)

