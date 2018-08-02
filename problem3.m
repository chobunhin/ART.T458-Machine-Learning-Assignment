clear; clc; close all;
%% generate test data
rng default
n = 100;
omega = randn(1, 1);
noise = 0.5 * randn(1,n);
x = randn(2, n);
y = 2 * (omega * x(1, :) + x(2, :) + noise > 0) - 1;
d = 2;
figure(2); hold on;
scatter(x(1,y>0)', x(2,y>0)', '.b');
scatter(x(1,y<0)', x(2,y<0)', '.r');

%% projected gradient method
lambda         = 1;
eta            = 0.005;
max_iter       = 1000;
alpha          = zeros(n,1);
vec_dual_lag   = zeros(max_iter, 1);
vec_hinge_loss = zeros(max_iter, 1);
% calculate kernel matrix K
K = diag(y) * x' * x * diag(y);

for t=1:max_iter
  % projected gradient step
  alpha_new = proj_box(alpha - 0.5*eta/lambda*K*alpha + eta, 0, 1);
  % calculate w
  w = 0.5/lambda * x * (alpha_new .* y');
  % calculate dual lagrange function
  vec_dual_lag(t) = -0.25/lambda * alpha_new' * K * alpha_new + sum(alpha_new);
  % calculate total hinge loss with regularization
  vec_hinge_loss(t) = sum(max(1 - (w'*x).*y, 0)) + lambda * norm(w,2)^2;
  
  alpha = alpha_new;
end
vec_dual_lag   = vec_dual_lag(1:t);
vec_hinge_loss = vec_hinge_loss(1:t);

figure(1);
plot(vec_dual_lag,'.b'); hold on;
plot(vec_hinge_loss,'.r'); hold on;
legend('dual Lagrange', 'total hinge loss');

figure(2);
draw_line_2points([0.5/w(1), 0], [0, 0.5/w(2)], 'g');