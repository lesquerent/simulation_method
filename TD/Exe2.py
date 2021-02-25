from import_packages import *


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
