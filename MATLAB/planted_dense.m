function [G,k, X0, m,n, Gc] = planted_dense(M, N,mmin, nmin,p, q)

% Generates a N-node graph with k clusters of size at least rmin.
% Start with all-ones blocks representing cliques indexed by each clusters and then
% delete clique edges, and add nonclique edges independently with probability p.

%% Initialize G:
G = rand(M, N);

% Add noise edge if Gij < p.
G = ceil(G - (1-q));



%% Determine block structure of V wrt r.

% # of clusters is at most N/rmin.
k = floor(min(N/nmin, M/mmin));

% Distribute nodes to clusters.
m = mmin*ones(k,1);
n = nmin*ones(k,1);

% Get remaining nodes to distribute
rmdrN = N - k*nmin
rmdrM = M - k*mmin
% Distribute nodes.
i = 1; % initialize index.

while rmdrN >= 1
n(i) = n(i)+1; % Update i-th entry of n.
rmdrN = rmdrN - 1; % Update rmdr.
% Update index.
if i == floor(0.75*k)
i=1;
else
i=i+1;
end
end

i = 1;
while rmdrM >= 1
m(i) = m(i)+1; % Update i-th entry of n.
rmdrM = rmdrM - 1; % Update rmdr.
% Update index.
if i == floor(0.75*k)
i=1;
else
i=i+1;
end
end


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
    X0(u_beg(i):(u_beg(i+1)-1), v_beg(i):(v_beg(i+1)-1)) = 1/sqrt(m(i)*n(i))*ones(m(i), n(i) );
end

