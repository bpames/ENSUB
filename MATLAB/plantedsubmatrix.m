function [A, X0, Y0] = plantedsubmatrix(M,m,p,q, symm)

% PLANTEDSUBMATRIX Makes binary matrix A with planted mn-submatrix.
%
% Generates mn-submatrix with expected density q in MxN matrix A with
% expected densities of remaining entries equal to q.
%
% INPUT:
%     M - desired dimensions of A. Scalar if symmetric.
%     m - desired dimensions of planted submatrix. Scalar if symmetric.
%     p - desired noise density.
%     q - desired in-group density.
%     symm - whether matrix should be symmetric. 
%
% OUTPUT:
%     A - matrix containing desired planted submatrix.
%     X0, Y0 - matrix representation of the planted submatrix.

if symm
    n = m;
else
    n = m(2); m = m(1);
end

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%  GENERATE NOISE ENTRIES OF A.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Initialize A as uniform random matrix.
% Round entries of A to 0 if less than 1-p and up to 1 otherwise.
tmp = rand(M);
if symm
    A = ceil(1/2*(tmp + tmp') -(1-p));
else
    A = ceil(tmp - (1 - p));
end

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%  FILL IN DENSE BLOCK.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Repeat with mn-block and threshhold 1-q.
if symm
    tmp = rand(m);
    A(1:m, 1:m) = ceil(1/2*(tmp + tmp') -(1-q));
else
    tmp = rand(m,n);
    A(1:m, 1:n) = ceil(tmp - (1 - q));
end
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%  CALCULATE MATRIX REPRESENTATION OF PLANTED SUBMATRIX.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% X0
X0 = zeros(M);
if symm
    X0(1:m,1:m) = ones(m);
else
    X0(1:m,1:n) = ones(m,n);
end


% Y0
Y0 = zeros(M);
if symm
    Y0(1:m,1:m) = ones(m) - A(1:m,1:m);
else
    Y0(1:m,1:n) = ones(m,n) - A(1:m,1:n);
end
