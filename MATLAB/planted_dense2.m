function [G,k, X0, M, N, Gc] = planted_dense2(m, n, p, q)

% Generates a N-node graph with k clusters of size at least rmin.
% Start with all-ones blocks representing cliques indexed by each clusters and then
% delete clique edges, and add nonclique edges independently with probability p.

%% Initialize G:
M = sum(m);
N = sum(n);
k = length(m);

G = rand(M, N);

% Add noise edge if Gij < p.
G = ceil(G - (1-q));



%% Determine block structure of V wrt r.

u_beg = ones(k,1); % Initialize positions of start point of each planted clique.
v_beg = ones(k,1);
% Set positions.
for i=2:k+1
    % Position is last position plus size of last clique.
   u_beg(i) = u_beg(i-1) + m(i-1); 
   v_beg(i) = v_beg(i-1) + n(i-1); 
end

%% Insert edges for planted dense subgraphs.

X0 = zeros(M,N); % Initialize X0.
Gc = zeros(M,N);

% Make dense blocks
for i=1:k
    % Get G.
    temp = rand(m(i),n(i));
    G(u_beg(i):(u_beg(i+1)-1), v_beg(i):(v_beg(i+1)-1)) = ceil(temp - (1-p));
    % Get Gc.
    Gc(u_beg(i):(u_beg(i+1)-1), v_beg(i):(v_beg(i+1)-1)) = G(u_beg(i):(u_beg(i+1)-1), v_beg(i):(v_beg(i+1)-1));
    % Get X0
    X0(u_beg(i):(u_beg(i+1)-1), v_beg(i):(v_beg(i+1)-1)) = ones(m(i), n(i) );
end

