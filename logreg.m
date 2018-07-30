function [ f ] = logreg( w, x, y)
%LOGREG: logistic regression.
%  w: parameter
%  x,y: training data (one pair)

f = log(1+exp(-1* y * (w' * x)));

end

