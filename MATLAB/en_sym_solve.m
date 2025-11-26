function [x,y,X,fval, time] = en_sym_solve(A, m, gamma, alpha, rho, tol, maxiter, quiet)
% Attempts to solve l1-regularized QP for densest submatrix using ADMM.

tic

if quiet == 0
    fprintf(' ********* iteration starts *********\n')
end

% input size
[M,~] = size(A);

% parameters
iter = 0;
% eM = ones(M,1);
% eN = ones(N,1);

rhohat = rho + 1 - alpha;

% Initialize primal variables.
mm = m(1);
mtilde = ceil(sqrt(2)*mm);
rsum = sum(A,1);
x = zeros(M, 1);
[~, rinds] = sort(rsum,"descend");
x(rinds(1:mtilde)) = mm/mtilde;
y = x;

% Initialize dual variables.
lambs = zeros(M,1);

% complement of A.
Abar = 1 - A;

% Iterate until maxits performed or converged.
while iter < maxiter
  
  % Update x via soft-thresholding.
  xold = x;
  x = soft_thresh((rho*y - lambs - Abar*y)/rhohat, gamma*alpha/rho);
  
  % Update y, u, v via projection onto probability simplex.
  yold = y;
  y = prob_simplex(x - (Abar*x - lambs)/rho, m);
  
  % Update lambda by dual gradient ascent.
  lambs = lambs + rho*(x - y);
  
  % check for termination
  pfeas = norm(x - y)/norm(x);
  dfeas = max(norm(xold - x)/norm(xold), norm(yold - y)/norm(yold));
  fval = gamma*(alpha*norm(x,1) + (1-alpha)/2*norm(x,2)^2) + 0.5*y'*A*y;

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


X = x*y';