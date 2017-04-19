function [alpha, b] = MTLSSVRTrain(trnX, trnY, trnN, gamma, lambda, p)
% 
% [alpha, b] = MTLSSVMTrain(trnX, trnY, trnN, gamma, lambda, p);
% 
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
% 
T = numel(trnN); 
NT = sum(trnN); 

Omega = Kerfun('rbf', trnX, trnX, p, 0) + eye(NT)/gamma; 

A = zeros(NT, T); 
for t = 1: T
    idx1 = sum(trnN(1: t-1)) + 1; 
    idx2 = sum(trnN(1: t)); 
    
    K = Kerfun('rbf', trnX(idx1: idx2, :), trnX(idx1: idx2, :), p, 0); 
    Omega(idx1: idx2, idx1: idx2) = Omega(idx1: idx2, idx1: idx2) + K*(T/lambda); 
    
    A(idx1: idx2, t) = ones(trnN(t), 1); 
end
% 
% alpha = [zeros(T), A'; A, Omega]\[ones(T, 1); trnY];
% 
% b = alpha(1: T); 
% alpha = alpha(T+1: end); 
eta = Omega \ A; 
nu = Omega \ trnY; 
S = A'*eta; 
b = inv(S)*eta'*trnY; 
alpha = nu - eta*b; 
