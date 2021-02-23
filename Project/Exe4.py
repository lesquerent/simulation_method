import numpy as np


def b_ary(nb_expansion, base):
    """

    Parameters
    ----------
    nb_expansion : array_like
        The array of expansion series
    base : int
        Base of the expansion

    Returns
    -------
    nb_expansion : array_like
         The nb expansion
    """
    list_index_to_increment = np.where(nb_expansion < base - 1)

    if len(list_index_to_increment[0] != 0):
        index_to_increment = list_index_to_increment[0][0]
        nb_expansion[0:index_to_increment] = 0
        nb_expansion[index_to_increment] += 1

    else:
        nb_expansion = nb_expansion * 0
        nb_expansion = np.append(nb_expansion, 1)
    return nb_expansion


def generate_k_b_ary_expansion(k, base=10):
    """
        Generate an array with k expansion
    :param k: int
        The number of value in the expansion
    :param base: int
        The base of calculation. Default : 10
    :return: array_like
        The array with the k expansion
    """
    nb_exp = np.array([0, 0, 0, 0])
    l = np.array([b_ary(nb_exp, base).copy() for i in range(k)])
    return np.array(l)


def van_der_corput_sequence(k, base):
    """
        Calculate the van der corput sequence of k number for the base selected
    :param k: int
        The number of value in the expansion
    :param base: int
        The base of calculation. Default : 10
    :return: list
        The list of the Van Der Corput Sequences
    """
    k_b_ary_expansion = generate_k_b_ary_expansion(k, base)
    list_res = []
    for elem in k_b_ary_expansion:
        n = len(elem)
        nb = 0
        for i in range(n):
            nb += elem[i] / base ** (i + 1)

        list_res.append(nb)
    return list_res


k = 10
base10 = 10
base2 = 2

test_b10 = van_der_corput_sequence(k, base10)
test_b2 = van_der_corput_sequence(k, base2)

print("The Van Der Corput Sequences with base 2 is :\n{}".format(test_b2))
print()
print("The Van Der Corput Sequences with base 10 is :\n{}".format(test_b10))
