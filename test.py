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

print(first_term_VDC(10,2))
