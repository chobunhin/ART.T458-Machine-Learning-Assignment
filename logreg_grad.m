function [ g ] = logreg_grad( w, x, y )
%LOGREG_GRAD: gradient of logistic regression.
%  w: parameter
%  x,y: training data (one pair)

z     = -1* y* (w' * x);
exp_z = exp(z);
wt    = exp_z / (1 + exp_z);
g = -1 * y * wt * x;

end

