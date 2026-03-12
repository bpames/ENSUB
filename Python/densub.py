def mat_shrink(Z, tau):
    '''
    MAT_SHRINK singular value soft-thresholding for nuclear norm prox fxn.
    INPUT:
        Z - matrix to have singular values thresholded.
        tau - threshold.
    OUTPUT:
        ZT - matrix following soft-thresholding.
    '''
    
    import numpy as np
    
    # Get dimensions of Z.
    [r,c] = Z.shape

    # Take SVD of Z.
    [U,S,V] = np.linalg.svd(Z) 
    # print("U", U.shape, "S", S.shape, "V", V[:, :r].shape)
    

    # Soft threshold singular values.
    s = np.maximum(S- tau, 0)
    
        
    # Reconstitute Z.
    if r < c:
        ZT = U @ np.diag(s) @ V[:r, ]        
    else:
        ZT = U[:, :c] @ np.diag(s) @ V
    
    return(ZT)
# End mat_shrink


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

class DenSub:
    """
    Class for solving the mn-submatrix problem using the alternating direction method of multipliers.

    Methods:
        constructor(tau, opt_tol, maxiters, verbose):
            define optimization parameters.
        solve(A,m,n,gamma): 
            solver for instance of problem with data matrix A, desired submatrix size (m,n), and regularization parameter gamma.
    Data attributes:
        tau: augmented Lagrangian penalty parameter.
        opt_tol: desired suboptimality stopping tolerance.
        maxiter: maximum number of iterations to perform.
        verbose: boolean, indicates whether to display iteration statistics.       
    """

    # Constructor.
    def __init__(self, tau=0.35, opt_tol=1e-4, maxiter=1000, verbose=False):
        self.tau = tau 
        self.opt_tol = opt_tol
        self.maxiter = maxiter
        self.verbose = verbose
    # end constructor.

    # Solver function.
    def solve(self, A, m, n, gamma):
        # imports
        from numpy import ones, zeros, maximum, minimum
        from numpy import sum as nsum
        from numpy import min as nmin
        from numpy import max as nmax
        from numpy.linalg import norm

        # Get dimensions of A.
        M,N = A.shape

        # Reciprocal of the augmented Lagrangian parameter.
        mu = 1/self.tau

        # Initial solutions.
        W = m*n / (M*N) * ones((M,N))
        X, Y, Z, Q = W, W, W, zeros((M,N))
        
        LambdaQ, LambdaZ, LambdaW = Q, Q, Q

        # Initial counter values.
        not_converged = True
        iters = 0

        ##########################################
        # Iteration statistics table.
        ##########################################
        if self.verbose:
            head_length=50        
            print("+"*head_length)
            print("DENSUB - ADMM for Densest Submatrix")
            print("+"*head_length)
            print(f"M={M:d}, N={N:d}, m={m:d}, n={n:d}, gamma={gamma:1.3e}")
            print("+"*head_length)
            print("It \t | Primal Gap \t | Dual Gap \t")
            print("+"*head_length)

        ##########################################
        # Iterative update via ADMM.
        ###########################################
        while not_converged:
            # Increment iteration counter.
            iters+=1            

            #++++++++++++++++++++++++++++++++++++++
            # Update Q.
            #++++++++++++++++++++++++++++++++++++++            
            Qold = Q # Save previous iterate.
            Q = X - Y + mu*LambdaQ # Update via dual ascent step.
            Q *= A # Project Q onto the support of A.
            # print("Q", Q)
            # fig, ax = plt.subplots(figsize=(4,4))
            # sns.heatmap(Q, cbar=False,
            #     linewidths=1,
            #     cmap="Purples")
            # plt.show()

            #++++++++++++++++++++++++++++++++++++++
            # Update X via matrix shrinkage.
            #++++++++++++++++++++++++++++++++++++++  
                      
            X = mat_shrink(1/3*(Y + Q + Z + W - mu*(LambdaQ + LambdaW + LambdaZ)), mu/3)
            # print("X", X)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Update Y via projection of residual onto nonnegative cone.
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            Y = maximum(X-Q-gamma*ones((M,N))*mu + LambdaQ*mu, 0)
            # print("Y", Y)
            #++++++++++++++++++++++++++++++++++++++
            # Update W.
            #++++++++++++++++++++++++++++++++++++++
            Wold = W 
            tempW = X + mu*LambdaW 

            # Scale and shift W so that the entries sum to m*n.
            alfa = (m*n - nsum(tempW))/(M*N)
            W = tempW + alfa*ones((M,N))
            # print("W", W)
            # print(nsum(W))
            

            #++++++++++++++++++++++++++++++++++++++
            # Update Z via clipping.
            #++++++++++++++++++++++++++++++++++++++
            
            Zold = Z 
            Ztemp = X + mu*LambdaZ 
            Z = minimum(maximum(Ztemp, 0), 1)

            # print("Z", Z)

            #++++++++++++++++++++++++++++++++++++++
            # Update dual variables by ascent step.
            #++++++++++++++++++++++++++++++++++++++
            
            LambdaQ = LambdaQ + self.tau*(X - Y - Q)
            LambdaW = LambdaW + self.tau*(X - W)             
            LambdaZ = LambdaZ + self.tau*(X - Z)
            # print("LQ", LambdaQ)
            # print("LW", LambdaW)
            # print("LZ", LambdaZ)

            #++++++++++++++++++++++++++++++++++++++
            # Check for convergence.            
            #++++++++++++++++++++++++++++++++++++++
            
            # Calculate primal residuals.
            NZ = norm(X-Z,'fro')
            NW = norm(X-W,'fro')
            NQ = norm(X-Y-Q, 'fro')            

            # Maximum normalized primal residual.
            errP = max([NZ, NW, NQ])/norm(X, "fro")

            # Calculate dual residuals.
            NDz = norm(Z - Zold, 'fro')
            NDw = norm((W-Wold), 'fro')
            NDp = norm((Q-Qold), 'fro')

            # Maximum relative dual residual.
            errD = max([NDz, NDw, NDp])/norm(X, 'fro')

            # Check for convergence.
            if (errP < self.opt_tol and errD < self.opt_tol):
                not_converged = False
                # Check if we have exceeded maximum number of iterations.
            elif iters >= self.maxiter:
                not_converged = False

            #++++++++++++++++++++++++++++++++++++++++++
            # Display iteration statistics if verbose.            
            #++++++++++++++++++++++++++++++++++++++++++
            
            if self.verbose: 
                if (iters % 5) == 0:
                    print("%3d \t | %1.3e \t | %1.3e" %(iters, errP, errD))          
        # End while loop.
        
        #++++++++++++++++++++++++++++++++++++++++++
        # Return solution at termination.
        #++++++++++++++++++++++++++++++++++++++++++
        return(X, Y, Q, iters)
    
    # end solver.
# end DenSub
    