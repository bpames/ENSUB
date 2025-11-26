function [u,v,X, fval, time] = ENSub(A, mm, gamma, alpha, rho, tol, maxiter, symm, quiet )
    if symm
        [u,v,X, fval, time] = en_sym_solve(A, mm, gamma, alpha, rho, tol, maxiter, quiet);
    else
        [u,v,X, fval, time] = en_solve(A, mm(1), mm(2), gamma, alpha, rho, tol, maxiter, quiet);
    end