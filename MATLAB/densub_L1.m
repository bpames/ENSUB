function [u,v,X,fval, time,  x, y] = densub_L1(A, m,n, gamma, rho, tol, maxiter, quiet)
% Attempts to solve l1-regularized QP for densest submatrix using ADMM.

tic

if quiet == 0
    fprintf(' ********* iteration starts *********\n')
end

% input size
[M,N] = size(A);

% parameters
iter = 0;
% eM = ones(M,1);
% eN = ones(N,1);

% Initialize primal variables.
u = rand(M,1); u = u*m/sum(u);
v = rand(N,1); v= v*n/sum(n);
y = [u;v]; x = y;
lambs = zeros(M+N,1);

% complement of A.
Abar = 1 - A;
Q = [zeros(M), Abar; Abar', zeros(N)];

% Iterate until maxits performed or converged.
while iter < maxiter
  
  % Update x via soft-thresholding.
  xold = x;
  x = soft_thresh(y - (lambs - Q*y)/rho, gamma);
  
  % Update y, u, v via projection onto probability simplex.
  yold = y;
  u = prob_simplex(x(1:M) - (Abar*x(M+1:M+N) + lambs(1:M))/rho, m);
  v = prob_simplex(x(M+1:M+N) - (Abar'*x(1:M) + lambs(M+1:M+N))/rho, n);
  y = [u;v];
  
  % Update lambda by dual gradient ascent.
  lambs = lambs + rho*(x - y);
  
  % check for termination
  pfeas = norm(x - y);
%   size(xold), size(x), size(yold), size(y)
  dfeas = max(norm(xold - x)/norm(xold), norm(yold - y)/norm(yold));
  fval = gamma*norm(x,1) + 0.5*x'*Q*x;

  % Printer statistics.
  if mod(iter,1) == 0 && iter > 0    && quiet ==0
    fprintf(' iter %4d   fval %10g  dfeas %10g pfeas %10g\n', iter, fval,  dfeas, pfeas)
  end

  % Check for convergence.
  if dfeas < tol && pfeas < tol
    break
  end
  
  % Update iteration counter.
  iter = iter + 1;
  
end

% Print end of information at termination.
if quiet == 0
    fprintf(' ********** termination ************\n')
    fprintf(' iter %4d   fval %10g  dfeas %10g  pfeas %10g\n', iter, fval, dfeas, pfeas)
end
time = toc;

if quiet == 0
    fprintf(' time elapsed %g\n', time)
end


X = u*v';