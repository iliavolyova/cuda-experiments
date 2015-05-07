import numpy as np

def power_method(A, x0, maxit):
    y = x0
    for i in xrange(maxit):
        y = A*y
        nrm = np.linalg.norm(y)
        print nrm
        y = y/nrm
    return y

def check(mat, power_eigvec):

    prd = mat*power_eigvec
    eigval = prd[0]/power_eigvec[0]
    print 'computed eigenvalue :' , eigval
    [eigs, vecs] = np.linalg.eig(mat)
    abseigs = list(abs(eigs))
    ind = abseigs.index(max(abseigs))
    print ' largest eigenvalue :', eigs[ind]

def main():
    dim = input("Dimenzije matrice:")

    arr = np.fromfile('A_' + str(dim) + '.dat', dtype=np.float64)
    A = arr.reshape(dim,dim)
    xarr = np.fromfile('x0_' + str(dim) + '.dat', dtype=np.float64)
    x0 = xarr.reshape(dim, 1)

    #print A, x0

    maxiter = input('Broj iteracija:')
    rndmat = np.matrix(A)
    rndvec = np.matrix(x0)
    eigmax = power_method(rndmat, rndvec, maxiter)
    check(rndmat, eigmax)

main()
