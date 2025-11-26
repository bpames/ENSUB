%% Random matrix.
% # Dimensions of the full matrix.
M=60; N = 50;

% # Dimensions of the dense block.
m=30; n = 20;

% # In-group and noise density.
q=0.75; p = 0.25;

% # Make binary matrix with planted mn-submatrix
[A, X0, Y0] = plantedsubmatrix(M,N,m,n,p,q);

imagesc([A,X0,Y0])

%% Make matrix/graph

% G = readmatrix("AG.csv");
% imagesc(G)

%% Solve using ENSub.
rho=0.75; 
opt_tol=1e-5;
quiet = false;
alpha=0.5;
mm=[30,20];
gamma = (q-p)*sqrt(mm(1)*mm(2))/10;
symm=false;
maxits=500;

[u,v, X] = ENSub(A, mm, gamma,  alpha, rho, opt_tol, ...
    maxits, symm, quiet);
imagesc([X, A])

%% =====================================================================

%% Make matrix/graph

% G = readmatrix("AG.csv");
% imagesc(G)

%% Attempt to solve using L1 problem.
mm = 40;
rho = 50;
gamma = mm/5;
tol = 1e-5;
maxiter = 2000;
quiet = false;
alpha = 0.65;
symm = true;


[u,v,X,fval, time] = ENSub(G, mm, gamma, alpha, rho, tol, maxiter, symm,quiet);

figure
imagesc([X, G])

% figure
% imagesc([u, v])

fprintf("norm A = %g\n", norm(G))
fprintf("gamma = %g \n", gamma*(1-alpha))


%% Attempt to solve using L1 problem.
mm = 40;
nn = 40;
rho = 50;
gamma = mm/5;
tol = 1e-5;
maxiter = 2000;
quiet = false;
alpha = 0.5;

[u,v,X,fval, time,  x, y] = densub_QP1(G, mm,nn, gamma, alpha, rho, tol, maxiter, quiet);

figure
imagesc([X, G])

