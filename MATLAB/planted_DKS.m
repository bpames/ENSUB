function [G, X0] = planted_DKS(N,K,p, q)

% Generates a N-node graph with containing planted clique of size k.
% Start with all-ones blocks representing cliques indexed by each clusters and then
% add clique edges with probability p, and add nonclique edges independently with probability q.

%% Initialize G:
G = rand(N); G = 0.5*(G + G');

% Add noise edge if Gij < p.
G = ceil(G - (1-q));


%% Insert edges for planted dense subgraph.

X0 = zeros(N); % Initialize X0.

% Make dense blocks
% Get G.
temp = rand(K); temp = 0.5*(temp + temp');
    G(1:K, 1:K) = ceil(temp - (1-p));

% Get X0
X0(1:K, 1:K) = ones(K);


