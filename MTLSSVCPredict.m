function [predictY, confusionMatrix]= MTLSSVCPredict(tstX, tstY, tstN, trnX, trnY, trnN, alpha, b, lambda, p)
% 
% [predictY, confusionMatrix]= MTLSSVCPredict(tstX, tstY, tstN, trnX, trnY,
% trnN, alpha, b, lambda, p); 
% 
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
% 
T = numel(trnN); 

K = Kerfun('rbf', tstX, trnX, p, 0) .* repmat(trnY', size(tstX, 1), 1); 
predictY = K * alpha; 
for t = 1: T
    tst_idx1 = sum(tstN(1: t-1)) + 1; 
    tst_idx2 = sum(tstN(1: t)); 
    
    trn_idx1 = sum(trnN(1: t-1)) + 1; 
    trn_idx2 = sum(trnN(1: t)); 
    
    predictY(tst_idx1: tst_idx2) =  predictY(tst_idx1: tst_idx2) + ...
        K(tst_idx1: tst_idx2, trn_idx1: trn_idx2)*alpha(trn_idx1: trn_idx2)*(T/lambda) + b(t);  
end

% sign() function
predictY(predictY >= 0) = +1; 
predictY(predictY <  0) = -1; 

% calcuate confusion matrix
TP = zeros(T, 1); 
FP = zeros(T, 1); 
TN = zeros(T, 1); 
FN = zeros(T, 1); 
for t = 1: T
    idx1 = sum(tstN(1: t - 1)) + 1; 
    idx2 = sum(tstN(1: t)); 
    TP(t) = sum((tstY(idx1: idx2) == +1) & (predictY(idx1: idx2) == +1)); 
    FP(t) = sum((tstY(idx1: idx2) == -1) & (predictY(idx1: idx2) == +1)); 
    TN(t) = sum((tstY(idx1: idx2) == -1) & (predictY(idx1: idx2) == -1)); 
    FN(t) = sum((tstY(idx1: idx2) == +1) & (predictY(idx1: idx2) == -1)); 
end
confusionMatrix = [TP,TN,FP,FN]; 
