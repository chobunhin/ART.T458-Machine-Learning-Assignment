clear; clc; close all;
%%
% rng default
m = 20;
n = 40;
r = 2;
Utrue = rand(m, r);
Vtrue = rand(n, r);
A = Utrue * Vtrue';
Afull = A;
ninc = 100; % number of not-valid-entries
Q = randperm(m * n, ninc);
A(Q) = NaN; % not valid entries


%% non-negative matrix factorization method

r_parm  = 3; % rank for non-negative matrix factorization
U = rand(m, r_parm);
Udelete = U;
V = rand(n, r_parm);
Vdelete = V;
Qmat    = ones(m, n);
Qmat(Q) = 0;

max_iter = 200;
err_NMF  = zeros(max_iter, 1);
for t = 1:max_iter
  for k=1:r_parm
    Udelete(:,k) = 0;
    Vdelete(:,k) = 0;
    uk = U(:,k);
    vk = V(:,k);
    % compute Rk
    Rk = A - Udelete*Vdelete';
    Rk(Q) = 0;
    % compute uk and vk
    vec_vk_norm2 = (repmat(vk', m, 1).* Qmat) * vk;
    uk = (Rk * vk) ./ vec_vk_norm2;
    uk = max(0, uk); % ensure nonnegativity
    
    vec_uk_norm2 = (repmat(uk', n, 1).* Qmat') * uk;
    vk = (Rk' * uk) ./ vec_uk_norm2;
    vk = max(0, vk); % ensure nonnegativity

    % update U,V
    U(:,k) = uk;
    V(:,k) = vk;
    Udelete(:,k) = uk;
    Vdelete(:,k) = vk;
  end
  err_NMF(t) = norm(Afull - U*V','fro')/norm(Afull,'fro');
  
end
err_NMF = err_NMF(1:t);
%% proximal gradient method on nuclear norm normalization
Z = zeros(m, n);
lambda = 0.02; % nuclear norm penalty parameter

max_iter_prox = 200;
err_PG = zeros(max_iter_prox, 1);

for t = 1: max_iter_prox
  % calculate gradient of smooth part
  g_Z    = 2 * (Z - A);
  g_Z(Q) = 0;
  % proximal gradient step
  Z_new = prox_nuclear(Z - 0.5 * g_Z, lambda);
  % update
  Z = Z_new;
  err_PG(t) = norm(Afull - Z,'fro')/norm(Afull,'fro');
end
err_PG = err_PG(1:t);
rank(Z)

% plot
figure(1);
semilogy(err_NMF, '.b'); hold on;
semilogy(err_PG, '.r');
legend('non-negative matrix fact','proximal gradient');






