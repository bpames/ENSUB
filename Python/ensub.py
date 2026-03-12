def plantedsubmatrix(M,N,m,n,p,q, seed="None"):
    '''
    PLANTEDSUBMATRIX Makes binary matrix A with planted mn-submatrix.
    
    Generates mn-submatrix with expected density q in MxN matrix A with
    expected densities of remaining entries equal to q.
    
    INPUT:
        M,N - desired dimensions of A.
        m,n - desired dimensions of planted submatrix.
        q - desired noise density.
        p - desired in-group density.
    
    OUTPUT:
        A - matrix containing desired planted submatrix.
        X0, Y0 - matrix representation of the planted submatrix.
    '''

    # Imports
    from numpy.random import default_rng
    from numpy import ceil, ones, zeros

    # Define RNG.
    if seed=="None":
        rng = default_rng()
    else:
        rng = default_rng(seed=seed)


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # GENERATE NOISE ENTRIES OF A.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Initialize A as uniform random matrix.
    tmp = rng.random((M,N))

    # Round entries of A to 0 if less than 1-p and up to 1 otherwise.
    A = ceil(tmp-(1-q))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # FILL IN DENSE BLOCK.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Repeat with mn-block and threshhold 1-q.
    tmp = rng.random((m,n))
    A[:m, :n] = ceil(tmp-(1-p))

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

def soft_thresh(a,t):
    from numpy import maximum as nmax
    return(nmax(0, a-t) - nmax(0,-a-t))

def prob_simplex(y0, k):
    from numpy import array, zeros, ones, sort, argsort, cumsum, concatenate, inf
    n = len(y0)
    
    
    if (k<0) or (k > n):
        raise ValueError("The sum constraint is infeasible!")
    
    if k==n:
        x = ones(n)
        return(x)
    elif k==0:
        x = zeros(n) 
        return(x)
    else:
        x = zeros(n) 
        idx = argsort(y0)
        y = y0[idx]

        s = cumsum(y)
        y = concatenate((y, array([inf])))
        
        # print(y)
        #+++++++++++++++++++++++++++++++++++
        # a = b = integers case.
        #+++++++++++++++++++++++++++++++++++
        if k==round(k):
            b=n-k
            # print(b)
            # print(y[b+1]-y[b])
            if y[b]-y[b-1]>=1:
                x[idx[b:]]=1
                return(x)
            # print(x)
    

        #+++++++++++++++++++++++++++++++++++
        # a = 0
        #+++++++++++++++++++++++++++++++++++       
        for b in range(n):
            # hypothesised gamma
            gamma = (k+b+1 - n - s[b])/(b+1)

            if ((y[0] + gamma) > 0) and ((y[b]+gamma) < 1) and ((y[b+1] + gamma) >= 1):
                # print(y[0:b])
                xtmp = concatenate((y[0:b+1] + gamma, ones(n-b-1)))
                x[idx]=xtmp
                return(x)
        
        #+++++++++++++++++++++++++++++++++++
        # a >= 1
        #+++++++++++++++++++++++++++++++++++
        for a in range(n):
            for b in range(a+1,n):
                # hypothesised gamma.
                gamma = (k + b + 1 - n + s[a] - s[b])/(b-a)

                if ((y[a]+gamma) <= 0) and ((y[a+1] + gamma) > 0) and ((y[b]+gamma)<1) and ((y[b+1] + gamma) >= 1):
                    
                    xtmp = concatenate((zeros(a+1), y[a+1:b+1] + gamma, ones(n-b-1)))
                    x[idx] = xtmp
                    return(x)                   

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
    def __init__(self, rho=1, alpha=0.5, opt_tol=1e-4, maxiter=1000, symmetric=False, verbose=False):
        self.rho = rho
        self.alpha = alpha
        self.opt_tol = opt_tol
        self.maxiter = maxiter
        self.symmetric = symmetric
        self.verbose = verbose
    # end constructor.

    # Solver function - unsymmetric case.
    def unsym_solve(self, A, mm, gamma):
        # imports
        from numpy import zeros, concatenate, block, ceil, sqrt, argsort
        from numpy import sum as nsum        
        from numpy.linalg import norm

        #++++++++++++++++++++++++++++++++++++++++++++
        # Initialisation.
        #++++++++++++++++++++++++++++++++++++++++++++++
        # Get dimensions of A.
        M,N = A.shape

        m,n = mm[0], mm[1]

        # Initial solutions.
        mtilde, ntilde = int(ceil(sqrt(2)*m)), int(ceil(sqrt(2)*n))
        csum, rsum = nsum(A,0), nsum(A,1)
        u, v = zeros(M), zeros(N)
        cinds, rinds = argsort(csum)[::-1], argsort(rsum)[::-1],
        u[rinds[0:mtilde]], v[cinds[0:ntilde]] = m/mtilde,  n/ntilde

        y = concatenate((u,v))
        x = y 
        lambs = zeros(M + N)

        # Make coefficient matrix Q.
        Abar = 1 - A
        Q = block([
            [zeros((M,M)), Abar],
            [Abar.T, zeros((N,N))]
        ])

        iters = 0
        not_converged=True
        rhohat = self.rho + 1 - self.alpha

        #++++++++++++++++++++++++++++++++++++++++++++++
        # Iterate until maxits performed or converged.
        #++++++++++++++++++++++++++++++++++++++++++++++
        # Iteration stats table.
        if self.verbose:
            head_length=55      
            print("+"*head_length)
            print("ENSUB - ADMM for Elastic Net Densest Submatrix")
            print("+"*head_length)
            print(f"M={M:d}, N={N:d}, m={m:d}, n={n:d}, gamma={gamma:1.3e}")
            print("+"*head_length)
            print("It \t | Objective \t | Primal Gap \t | Dual \t")
            print("+"*head_length)

        

        while not_converged and (iters < self.maxiter):
            # Increment iteration counter.
            iters+=1  

            # Update x via soft-thresholding.
            xold = x             
            x = soft_thresh(
                a=(self.rho*y - lambs - Q @ y)/rhohat, 
                t=gamma*self.alpha/self.rho
            )

            # Update u,v,y via projection onto the capped probability simplex.
            yold = y 
            u = prob_simplex(
                y0= x[:M] - (Abar @ x[M:] + lambs[:M])/self.rho, 
                k = m 
            )
            v = prob_simplex(
                y0= x[M:] - (Abar.T @ x[:M] + lambs[M:])/self.rho,
                k = n
            )
            # print(u.shape)
            # print(v.shape)
            y = concatenate((u,v))

            # Update lambda by dual gradient ascent. 
            lambs = lambs + self.rho*(x - y) 

            # Check for convergence. 
            pfeas = norm(x - y)/norm(x) 
            dfeas = max(
                norm(xold - x)/norm(xold), 
                norm(yold - y)/norm(yold)
            )
            fval = gamma*(self.alpha*norm(x,1) + (1-self.alpha)/2*norm(x,2)**2) + 0.5*y.T @ Q @ y 

            if dfeas < self.opt_tol and pfeas < self.opt_tol: # CONVERGED!!
                not_converged = False

            #++++++++++++++++++++++++++++++++++++++++++
            # Display iteration statistics if verbose.            
            #++++++++++++++++++++++++++++++++++++++++++
            
            if self.verbose:                 
                print("%3d \t | %1.3e \t | %1.3e \t | %1.3e" %(iters, fval, pfeas, dfeas))

        return(u,v, x, y, fval, iters)
    # End Unsymmetric_Solve

    # Symmetric solver.
    def sym_solve(self, A, m, gamma):
        # imports
        from numpy import zeros, ceil, sqrt, argsort
        from numpy import sum as nsum        
        from numpy.linalg import norm

        # Get size of A.
        M = A.shape[0]

        # Initial solutions.
        mtilde = int(ceil(sqrt(2)*m))
        rsum = nsum(A,0)
        x = zeros(M)
        rinds = argsort(rsum)[::-1]
        x[rinds[0:mtilde]] = m/mtilde

        y = x
        lambs = zeros(M)

        # Coefficient matrix.
        Abar = 1 - A

        # Parameters.
        iters = 0
        not_converged=True
        rhohat = self.rho + 1 - self.alpha

        #++++++++++++++++++++++++++++++++++++++++++++++
        # Iterate until maxits performed or converged.
        #++++++++++++++++++++++++++++++++++++++++++++++
        # Iteration stats table.
        if self.verbose:
            head_length=55      
            print("+"*head_length)
            print("ENSUB - ADMM for Elastic Net Densest Submatrix")
            print("+"*head_length)
            print("Symmetric -- " + f"M={M:d}, m={m:d}, gamma={gamma:1.3e}")
            print("+"*head_length)
            print("It \t | Objective \t | Primal Gap \t | Dual \t")
            print("+"*head_length)        

        while not_converged and (iters < self.maxiter):
            # Increment iteration counter.
            iters+=1  

            # Update x via soft-thresholding.
            xold = x   
            x = soft_thresh(
                a=(self.rho*y - lambs - Abar @ y)/rhohat, 
                t=gamma*self.alpha/self.rho
            )

            # Update y via projection onto the capped probability simplex.
            yold = y 
            y = prob_simplex(
                y0= x + (-Abar @ x + lambs)/self.rho, 
                k = m 
            )

            # Update lambda by dual gradient ascent. 
            lambs = lambs + self.rho*(x - y) 

            # Check for convergence. 
            pfeas = norm(x - y)/norm(x) 
            dfeas = max(
                norm(xold - x)/norm(xold), 
                norm(yold - y)/norm(yold)
            )
            fval = gamma*(self.alpha*norm(x,1) + (1-self.alpha)/2*norm(x,2)**2) + x.T @ Abar @ x / 2 

            if dfeas < self.opt_tol and pfeas < self.opt_tol: # CONVERGED!!
                not_converged = False

            #++++++++++++++++++++++++++++++++++++++++++
            # Display iteration statistics if verbose.            
            #++++++++++++++++++++++++++++++++++++++++++
            if self.verbose:                 
                print("%3d \t | %1.3e \t | %1.3e \t | %1.3e" %(iters, fval, pfeas, dfeas))

        return(x, y, fval, iters)
    # End symmetric solver.

    # General solve function.
    def solve(self, A, mm, gamma):
        if self.symmetric:
            return(self.sym_solve(A, mm, gamma))
        else:
            return(self.unsym_solve(A, mm, gamma))

# End ENSub class

            