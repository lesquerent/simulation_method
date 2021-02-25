from import_packages import *


def box_Muller():
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2


def modified_Box_Muller():
    z = 1
    u1, u2 = 0, 0

    # Rejection
    while (z >= 1):
        u1 = np.random.uniform(-1, 1)
        u2 = np.random.uniform(-1, 1)

        z = (u1 ** 2) + (u2 ** 2)

    # Acceptance
    r = -2 * np.log(z)

    x = np.sqrt(r / z) * u1
    y = np.sqrt(r / z) * u2

    return x, y

# Exercise 1 Box Muller
# bx = box_Muller()
# print(bx[0])
# print(bx[1])

# mbx = modified_Box_Muller()
# print(mbx[0])
# print(mbx[1])