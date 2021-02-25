from import_packages import *


def z_distribution():
    """

    Returns
    -------
        :type array
        Value of z following the positive part of a normal N(0,1) distribution
        Number of iteration necessary to generate z
    """
    nb_of_try = 1
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)

    while u1 > np.exp(-(1 / 4) * (u2 / u1) ** 2):
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        nb_of_try += 1

    return np.array([[u2/u1], [nb_of_try]])


# Generation of z values
z_nb_try = np.array([z_distribution() for i in range(100000)])
z_value = z_nb_try[:, 0]

# Number of try for the z distribution
nb_try = z_nb_try[:, 1]

# Acceptance rate
acceptance_rate = sum(nb_try) / len(z_value)
print('The acceptance rate for the z distribution is {}'.format(float(acceptance_rate)))
# results : The acceptance rate for the z distribution is 1.59944

# Generation of normal values
normal_value = np.random.normal(0, 1, 100000)

# Plot
fig, ax = plt.subplots()
ax.hist(normal_value, bins=50, histtype='step', label=['Normal distribution'])
ax.hist(z_value, bins=50, label='Z distribution')
ax.legend(loc='best', shadow=True)
plt.title('Distribution of Z')
plt.show()
