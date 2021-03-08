from import_packages import *


def bv_comonotonicity_copula(u1, u2):
    return np.minimum(u1, u2)


def bv_cuntermonotonicity_copula(u1, u2):
    return np.maximum(u1 + u2 - 1, 0)


def bv_gumbel_copula(u1, u2, theta):
    return np.exp(-((-np.log(u1)) ** theta + (-np.log(u2)) ** theta) ** (1 / theta))


def bv_clayton_copula(u1, u2, theta):
    return (u1 ** -theta + u2 ** -theta - 1) ** (-1 / theta)


def bv_gauss_copula(u1, u2, theta):
    return (1 / (np.pi * np.sqrt(1 - theta ** 2))) * np.exp(
        (-1 / (2 * (1 - theta ** 2))) * ((u1 ** 2) - 2 * theta * u1 * u2 + (u2 ** 2)))


def simu_gauss_copula(number_of_value = 1000):
    x1 = np.random.normal(0, 1, size=number_of_value)
    x2 = np.random.normal(0, 1, size=number_of_value)
    u = np.array([stats.norm.cdf(x1), stats.norm.cdf(x2)])
    return u.T
    


def question1():
    THETA = 2
    u1_ = np.linspace(0, 1, 100)
    u2_ = np.linspace(0, 1, 100)

    u1_, u2_ = np.meshgrid(u1_[1:], u2_[1:])

    dict_of_copula = dict()
    dict_of_copula['Clayton Copula bivariate copula'] = bv_clayton_copula(u1_, u2_, THETA)
    dict_of_copula['Gumbel Copula bivariate copula'] = bv_gumbel_copula(u1_, u2_, THETA)
    dict_of_copula['Comonotonicity bivariate copula'] = bv_comonotonicity_copula(u1_, u2_)
    dict_of_copula['Countermonotonicity bivariate copula'] = bv_cuntermonotonicity_copula(u1_, u2_)
    # dict_of_copula[' Gauss bivariate copula'] = bv_gauss_copula(u1_, u2_, THETA)

    # Display plot
    for key, value in dict_of_copula.items():
        fig = plt.figure()
        fig.suptitle(key)

        ax = fig.add_subplot(2, 1, 1)
        ax.contour(u1_, u2_, value, 20, cmap='RdGy')
        ax.set_title('Contour')

        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.plot_surface(u1_, u2_, value, cmap='RdGy')
        ax.set_title('Surface')

    plt.show()

if __name__ == '__main__':
    # question1()

    print(simu_gauss_copula(10))
