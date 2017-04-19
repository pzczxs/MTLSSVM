function [predictY, TSE, R2]= MTLSSVRPredict(tstX, tstY, tstN, trnX, trnN, alpha, b, lambda, p)
% 
% [predictY, TSE, R2]= MTLSSVRPredict(tstX, tstY, tstN, trnX, trnN, alpha, 
% b, lambda, p);  
% 
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
% 
T = numel(trnN); 

K = Kerfun('rbf', tstX, trnX, p, 0); 
predictY = K * alpha; 
for t = 1: T
    tst_idx1 = sum(tstN(1: t-1)) + 1; 
    tst_idx2 = sum(tstN(1: t)); 
    
    trn_idx1 = sum(trnN(1: t-1)) + 1; 
    trn_idx2 = sum(trnN(1: t)); 
    
    predictY(tst_idx1: tst_idx2) =  predictY(tst_idx1: tst_idx2) + ...
        K(tst_idx1: tst_idx2, trn_idx1: trn_idx2)*alpha(trn_idx1: trn_idx2)*(T/lambda) + b(t);  
end

% calculate Total Squared Error and squared correlation coefficient
TSE = zeros(1, T); 
R2 = zeros(1, T); 
for t = 1: T
    tst_idx1 = sum(tstN(1: t-1)) + 1; 
    tst_idx2 = sum(tstN(1: t)); 
    
    TSE(t) = sum((predictY(tst_idx1: tst_idx2) - tstY(tst_idx1: tst_idx2)).^2); 
    R = corrcoef(predictY(tst_idx1: tst_idx2), tstY(tst_idx1: tst_idx2)); 
    if size(R, 1) >  1
        R2(t) = R(1, 2)^2; 
    end
end
