function [alpha, b] = MTLSSVCTrain(trnX, trnY, trnN, gamma, lambda, p)
% 
% [alpha, b] = MTLSSVCTrain(trnX, trnY, trnN, gamma, lambda, p);
% 
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
% 
T = numel(trnN); 
NT = sum(trnN); 

Omega = Kerfun('rbf', trnX, trnX, p, 0) .* (trnY*trnY') + eye(NT)/gamma; 

A = zeros(NT, T); 
for t = 1: T
    idx1 = sum(trnN(1: t-1)) + 1; 
    idx2 = sum(trnN(1: t)); 
    
    K = Kerfun('rbf', trnX(idx1: idx2, :), trnX(idx1: idx2, :), p, 0) .* ...
        (trnY(idx1: idx2)*trnY(idx1: idx2)'); 
    Omega(idx1: idx2, idx1: idx2) = Omega(idx1: idx2, idx1: idx2) + K*(T/lambda); 
    
    A(idx1: idx2, t) = trnY(idx1: idx2); 
end
% 
% alpha = [zeros(T), A'; A, Omega]\[zeros(T, 1); ones(NT, 1)];
% b = alpha(1: T); 
% alpha = alpha(T+1: end); 
eta = Omega \ A; 
nu = Omega \ ones(NT, 1); 
S = A'*eta; 
b = inv(S)*eta'*ones(NT, 1); 
alpha = nu - eta*b; 
