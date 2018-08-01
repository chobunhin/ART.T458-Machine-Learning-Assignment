function [ prox_X ] = prox_nuclear( X, lambda )
% PROX_NUCLEAR calculates proximal operator of matrix nuclear norm:
% prox_{lambda*||.||_*}(X)

[U,S,V] = svd(X,'econ');
S = diag(max(diag(S)-lambda,0));
prox_X = U*S*V';

end

