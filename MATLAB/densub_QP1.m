function [u,v,X,fval, time,  x, y] = densub_QP1(A, m,n, mu, rho, tol, maxiter, quiet)
% Attempts to solve l1-regularized QP for densest submatrix.

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
u = rand(M,1); v = rand(N,1);
x1 = u; x2 = v;
y1 = u; y2 = v;

% Initialize dual variables.
lambs1 = zeros(M,1); phis1 = lambs1;
lambs2 = zeros(N,1); phis2 = lambs2;

% complement of A.
Abar = 1 - A;

% Iterate until maxits performed or converged.
while iter < maxiter
  
  % Update u.
  uold = u;
  u = (x1 + y1 + 1/rho*(-lambs1 - phis1 - Abar*v - mu))/2;

  % Update v.
  vold = v;
  v = (x2 + y2 + 1/rho*(-lambs2 - phis2 - Abar'*u -  mu))/2;
    
  % Update x.
  xold = [x1; x2];
  tmp1 = u - lambs1/rho; tmp2 = v - lambs2/rho;
  x1 = tmp1 + rho/M*(m - sum(tmp1));
  x2 = tmp2 + rho/N*(n - sum(tmp2));
  x = [x1;x2];
  
  % Update y.
  yold = [y1; y2];
  y1 = min(1, max(u - phis1/rho, 0));
  y2 = min(1, max(v - phis2/rho, 0));
  y = [y1;y2];

  % Dual gradient step.
  lambs1 = lambs1 + rho*(x1 - u);
  phis1 = phis1 + rho*(y1 - u);
  lambs2 = lambs2 + rho*(x2 - v);
  phis2 = phis2 + rho*(y2 - v);
  
  % check for termination
  pfeas = max([-min(u), -min(v), 1 - max(u), 1 - max(v), abs(sum(u) - m), abs(sum(v) - n)]);
  dfeas = max([norm(uold - u)/norm(uold), norm(vold - v)/norm(vold), ...
      norm(xold - x)/norm(xold), norm(yold - y)/norm(yold)]);

  fval = mu*(sum(u) + sum(v)) + u'*Abar*v;

  %M1 = W(1:m,1:m)+U(1:m,1:m);
  %M2 = W(m+1:N,m+1:N) + U(m+1:N,m+1:N) ;
%   dval = 2*k*min(eig(W+U)) - sum(sum(X.*U)); % asymptotically converging to dual optimal value, not necessarily a lower bound.
  %dval = k*(min(eig(M1)) + min(eig(M2))) - sum(sum(X.*U)); % asymptotically converging to dual optimal value, not necessarily a lower bound.
%   relgap = abs(fval - dval)/max(abs(fval),1); % relative duality gap
  if mod(iter,25) == 0 && iter > 0    && quiet ==0
    fprintf(' iter %4d   fval %10g  dfeas %10g pfeas %10g\n', iter, fval,  dfeas, pfeas)
  end
  if dfeas < tol && pfeas < tol
    break
  end
  
  iter = iter + 1;
  
end

if quiet == 0
    fprintf(' ********** termination ************\n')
    fprintf(' iter %4d   fval %10g  dfeas %10g  pfeas %10g\n', iter, fval, dfeas, pfeas)
end
time = toc;

if quiet == 0
    fprintf(' time elapsed %g\n', time)
end


X = u*v';