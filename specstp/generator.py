import numpy as np

def to_file(A, x0, dim):
    A_file = open('A_' + str(dim) + '.dat', 'wb')
    A.tofile(A_file)
    A_file.close()

    x0_file = open('x0_' + str(dim) + '.dat', 'wb')
    x0.tofile(x0_file)
    x0_file.close()

def main():
    dim = input('Give the dimension : ')

    D = np.matrix(np.diag(np.random.normal(0, 1, dim)))
    q, r = np.linalg.qr(np.random.normal(0, 1, (dim,dim)))
    q = np.matrix(q)
    A = q * D * q.T
    x0 = np.random.normal(0, 1, (dim, 1)).astype(np.float64)

    to_file(A, x0, dim)

main()