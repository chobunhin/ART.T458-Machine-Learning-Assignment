clear; clc; 
figure(1);
hold off;

%% problem setup
rng default
n = 40;
omega = randn(1, 1);
noise = 0.5 * randn(1,n);
x = randn(2, n);
y = 2 * (omega * x(1, :) + x(2, :) + noise > 0) - 1;
d = 2;
figure(2); hold off;
scatter(x(1,y>0)', x(2,y>0)', '.b');hold on;
scatter(x(1,y<0)', x(2,y<0)', '.r');hold on;
%% batch steepest descent
% init
w   = zeros(2, 1);
w   = [1;0];
batch_size = 10;

stop_ratio = 0.0001;
lambda = 0.1;
lipschitz = norm(reshape(x, 2*n,1))^2 * 0.25 + 2 * lambda;
step_size = 0.1/lipschitz;
max_t = 500;

f  = logreg_full(w, x, y, lambda);
f0 = f;
for t = 1:max_t
  indices = randperm(n);
  indices = indices(1:batch_size);
  dir = zeros(2, 1);
  for i = 1:batch_size
    dir = dir + logreg_grad(w, x(:, indices(i)), y(indices(i)));
  end
  dir = dir + 2 * lambda * w;
  w_new = w - step_size * dir;
  
  f_new = logreg_full(w_new, x, y, lambda);
  
  figure(1);
  plot(t, f, '.b'); hold on;
  
  w = w_new;
  f = f_new;
end
% plot result by batch steep descent
w_batch = w;
figure(2);
draw_line_2points([0.5/w_batch(1), 0], [0, 0.5/w_batch(2)], 'k');

%% AdaGrad method
w          = zeros(2, 1);
f          = logreg_full(w, x, y, lambda);
f_grad0    = logreg_full_grad( w, x, y, lambda);
f_sumgrad  = zeros(size(f_grad0));
f_sumgrad2 = zeros(size(f_grad0));

eta0       = 0.005; % for step size control
delta      = 0.000001; % for diagonal hessian regularization
for t = 1:max_t
  f_grad     = logreg_full_grad( w, x, y, lambda);
  f_sumgrad  = f_sumgrad + f_grad;
  f_sumgrad2 = f_sumgrad2 + f_grad .* f_grad;
  f_diaghess = sqrt(f_sumgrad2) + delta;
  dir = f_sumgrad ./ f_diaghess;
  w_new = w - eta0 * dir;
  f_new = logreg_full(w_new, x, y, lambda);
  if f - f_new < stop_ratio * abs(f0)
    break;
  end
  figure(1);
  plot(t, f, '.r'); hold on;
  
  w = w_new;
  f = f_new;
end

% plot result by adagrad method
figure(2);
draw_line_2points([0.5/w(1), 0], [0, 0.5/w(2)], 'g');
legend('y=1', 'y=-1','batch', 'adagrad');