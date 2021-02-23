import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from TD1 import *

spot_price = 100
strike_price = 150
maturity = 1.0
volatility = 0.2
risk_free_rate = 0.01

# Exercise 3 MC and BS
nb_sequences = 100000
nb_price = 1
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



mu = 0.3
new_call_option_priced_with_MC_method = pricing_option_with_MC_method(spot_price, strike_price, volatility, maturity,
                                                                  mu,
                                                                  nb_sequences, nb_price, type_option)

new_all_price = new_call_option_priced_with_MC_method[1]
new_mean_payoff = new_call_option_priced_with_MC_method[2]
new_option_price = new_call_option_priced_with_MC_method[0]
new_array_of_payoff = new_call_option_priced_with_MC_method[3]

plt.hist(all_price[:, -1], 100)
plt.hist(new_all_price[:, -1], 100)
plt.show()

S_T = all_price[:, -1]
new_S_T = new_all_price[:, -1]

radon_nikodym_der = np.exp(-(np.log(new_S_T/spot_price) - (risk_free_rate - volatility*volatility/2)*maturity)**2/(2*volatility*volatility*maturity)
                           +(np.log(new_S_T)))
