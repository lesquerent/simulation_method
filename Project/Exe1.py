import numpy as np
import matplotlib.pyplot as plt


def z_distribution():
    """

    Returns
    -------
        Value of z following the positive part of a normal N(0,1) distribution
    """
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)

    while u1 > np.exp(-(1 / 4) * (u2 / u1) ** 2):
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)

    return u2 / u1


# Generation of z values
z_value = [z_distribution() for i in range(100000)]

# Generation of normal values
normal_value = np.random.normal(0, 1, 100000)

# Plot
fig, ax = plt.subplots()
ax.hist(normal_value, bins=50, histtype='step', label=['Normal distribution'])
ax.hist(z_value, bins=50, label='Z distribution')
ax.legend(loc='best', shadow=True)
plt.title('Distribution of Z')
plt.show()
