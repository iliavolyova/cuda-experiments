import numpy as np

def main():
    dim = input('Give the dimension : ')

    A = np.random.normal(0, 1, (dim, dim))
    x0 = np.random.normal(0, 1, (dim, 1)).astype(np.float64)

    A_file = open('A_' + str(dim) + '.dat', 'wb')
    A.tofile(A_file)
    A_file.close()

    x0_file = open('x0_' + str(dim) + '.dat', 'wb')
    x0.tofile(x0_file)
    x0_file.close()

main()