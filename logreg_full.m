function [ f ] = logreg_full( w, x, y, lambda)
%LOGREG_FULL returns full logreg target function with L2 regularization

f = sum(log(1+exp(-1* y.* (w' * x)))) ;
f = f + lambda * norm(w,2)^2;


end

