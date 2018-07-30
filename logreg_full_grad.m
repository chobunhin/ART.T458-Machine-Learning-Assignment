function [ g ] = logreg_full_grad( w, x, y, lambda)
%LOGREG_FULL_GRAD returns gradient of 
% full logreg target function with L2 regularization
n = length(y);
g = zeros(size(w));

for i=1:n
  g = g + logreg_grad( w, x(:,i), y(i) );
end
g = g + 2 * lambda * w;

end

