import numpy as np


def b_ary(nb_expansion, base):
    """

    Parameters
    ----------
    nb_expansion : array_like
        The array of expansion serie
    base : int
        Base of the expansion

    Returns
    -------
    nb_expansion : array_like
         The nb expansion
    """
    nb_expansion += 1
    nn = np.where(nb_expansion < base)[0][0]
    nb_expansion[0:nn] = 0
    nb_expansion[nn + 1:] -= 1
    return nb_expansion


nb_exp = np.array([9, 9, 2, 1])
base_ = 10

nb_exp = b_ary(nb_exp, base_)
print(nb_exp)


def van_der_corput(nb, base):
    l = []
    while nb > 0:
        q = nb % base
        l.append(q)
        nb = nb // base

    l.reverse()
    return l


def first_term_VDC(n, base):
    list_vdc = []
    for i in range(1, n + 1):
        x = van_der_corput(i, base)
        var = 0
        n = len(x)
        for i in range(n):
            var += x[i] / base ** (n - i)
        list_vdc.append(var)
        return list_vdc
