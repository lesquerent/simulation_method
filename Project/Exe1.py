from import_packages import *
from scipy.stats import norm
from math import pi

def z_distribution():
    """
        Generate one value following the z distribution and
        the number of try to get this value
    Returns
    -------
        :type numpy.array
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
ax.hist(z_value, bins=50, label='Z distribution ', color = "orange")
ax.hist(normal_value, bins=50, histtype='step', label=['Normal distribution'])

ax.legend(loc='best', shadow=True)
plt.title('Distribution of Z compare to a normal N(0,1) distribution')
plt.show()


# Z theorique
index = np.linspace(-7, 7, 100)
z_theoretical = (1 / 2) * np.exp(-(1 / 4) * index ** 2)
index_pos = np.linspace(0, 7, 100)

z_theoretical_pos = (1 / 2) * np.exp(-(1 / 4) * index_pos ** 2)

#Normal law

normal_value_pdf = (1/np.sqrt(2*pi))*np.exp((-1/2)*index**2)

#plot

fig2, ax2 = plt.subplots()
#ax2.plot(index, z_theoretical, color ='yellow', label ='Theoretical z beetwen -inf and +inf')
ax2.plot(index_pos, z_theoretical_pos, color ='red', label ='Theoretical z between 0 and +inf')
ax2.plot(index,normal_value_pdf, color = 'blue', label = 'Normal N(0,1)')
ax2.legend(loc='best', shadow=True)
plt.title('Distribution of Z theoretical compare to a normal N(0,1) distribution')
plt.show()