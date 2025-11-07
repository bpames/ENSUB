def plantedsubmatrix(M,N,m,n,p,q):
    '''
    PLANTEDSUBMATRIX Makes binary matrix A with planted mn-submatrix.
    
    Generates mn-submatrix with expected density q in MxN matrix A with
    expected densities of remaining entries equal to q.
    
    INPUT:
        M,N - desired dimensions of A.
        m,n - desired dimensions of planted submatrix.
        p - desired noise density.
        q - desired in-group density.
    
    OUTPUT:
        A - matrix containing desired planted submatrix.
        X0, Y0 - matrix representation of the planted submatrix.
    '''

    # Imports
    from numpy.random import rand
    from numpy import ceil, ones, zeros

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # GENERATE NOISE ENTRIES OF A.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Initialize A as uniform random matrix.
    tmp = rand(M,N)

    # Round entries of A to 0 if less than 1-p and up to 1 otherwise.
    A = ceil(tmp-(1-p))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # FILL IN DENSE BLOCK.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Repeat with mn-block and threshhold 1-q.
    tmp = rand(m,n)
    A[:m, :n] = ceil(tmp-(1-q))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # CALCULATE MATRIX REPRESENTATION OF PLANTED SUBMATRIX.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # X0
    X0 = zeros((M,N))
    X0[:m,:n] = ones((m,n))

    # Y0
    Y0 = zeros((M,N))
    Y0[:m,:n] = ones((m,n)) - A[:m,:n]
    
    # Return A, X0, Y0.
    return(A, X0, Y0)
# end plantedsubmatrix.

class ENSub:
    """
    Class for solving the mn-submatrix problem using the alternating direction method of multipliers for solving non-convex QP relaxation.

    Methods:
        constructor(tau, opt_tol, maxiters, verbose):
            define optimization parameters.
        solve(A,m,n,gamma): 
            solver for instance of problem with data matrix A, desired submatrix size (m,n), and regularization parameter gamma.
    Data attributes:
        rho: augmented Lagrangian penalty parameter.
        opt_tol: desired suboptimality stopping tolerance.
        maxiter: maximum number of iterations to perform.
        verbose: boolean, indicates whether to display iteration statistics.       
    """

     # Constructor.
    def __init__(self, rho=0.35, opt_tol=1e-4, maxiter=1000, verbose=False):
        self.rho = rho
        self.opt_tol = opt_tol
        self.maxiter = maxiter
        self.verbose = verbose
    # end constructor.

    # Solver function.
    def solve(self, A, m, n, gamma):
        # imports
        from numpy import ones, zeros, maximum, minimum, concatenate
        from numpy import sum as nsum
        from numpy import min as nmin
        from numpy import max as nmax
        from numpy.linalg import norm
        from numpy import concatenate

        # Get dimensions of A.
        M,N = A.shape

        # Initial solutions.
        u0 = sum(A,1)
        u = m*u0/sum(u0)
        v0 = sum(A,0)
        v = n*v0/sum(v0)

        y = concatenate(u,v)
        x = y 
        lambs = zeros(M + N)

        # Make coefficient matrix Q.
        Abar = 1 - A
        Q = zeros()