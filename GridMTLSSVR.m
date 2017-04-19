function [gamma_best, lambda_best, p_best, MSE_best, MSE_all, R2] = GridMTLSSVR(trnX, trnY, trnN, fold, gamma_best, lambda_best, p_best, MSE_best)
% 
% [gamma_best, lambda_best, p_best, MSE_best, R2] = GridMTLSSVR(trnX, trnY,
% gamma_best, lambda_best, p_best, MSE_best); 
% 
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
% 
gamma = 2.^(-5: 2: 15); 
lambda = 2.^(-10: 2: 10); 
p = 2.^(-15: 2: 3);

trnN = trnN(:); 
task_start = cumsum(trnN); 
task_start = [1; task_start(1: end -1) + 1]; 
T = numel(task_start); 
task_end = [task_start(2: end) - 1; numel(trnY)]; 

% random permutation
for t = 1: T
    [trnX(task_start(t): task_end(t), :), trnY(task_start(t): task_end(t))] = ...
        random_perm(trnX(task_start(t): task_end(t), :), trnY(task_start(t): task_end(t))); 
end

MSE = zeros(fold, T); 
curR2 = zeros(1, T); 
R2 = zeros(1, T); 
for i = 1: length(gamma)
    for j = 1: length(lambda)
        for k = 1: length(p)
            predictY = cell(1, T); 
            
            for v = 1: fold
                train_inst = []; 
                train_lbl = []; 
                test_inst = []; 
                test_lbl = []; 
                trnN = zeros(1, T); 
                tstN = zeros(1, T); 
                for t = 1: T
                    [tr_inst, tr_lbl, ts_inst, ts_lbl] = ...
                        folding(trnX(task_start(t): task_end(t), :), trnY(task_start(t): task_end(t)), fold, v);
                    train_inst = [train_inst; tr_inst]; 
                    train_lbl = [train_lbl; tr_lbl]; 
                    test_inst = [test_inst; ts_inst]; 
                    test_lbl = [test_lbl; ts_lbl]; 
                    
                    trnN(t) = numel(tr_lbl); 
                    tstN(t) = numel(ts_lbl); 
                end
                
                [alpha, b] = MTLSSVRTrain(train_inst, train_lbl, trnN, gamma(i), lambda(j), p(k));
                [tmpY, MSE(v, :)]= MTLSSVRPredict(test_inst, test_lbl, tstN, train_inst, trnN, alpha, b, lambda(j), p(k)); 
                
                for t = 1: T
                    idx1 = sum(tstN(1: t-1)) + 1; 
                    idx2 = sum(tstN(1: t)); 
                    
                    predictY{t} = [predictY{t}; tmpY(idx1: idx2)]; 
                end
            end
            
            curMSE = sum(sum(MSE)) / numel(trnY); 
            for t = 1: T
                R = corrcoef(predictY{t}, trnY(task_start(t): task_end(t))); 
                curR2(t) = R(1, 2)^2; 
            end
            
            if MSE_best > curMSE
                gamma_best = gamma(i); 
                lambda_best = lambda(j); 
                p_best = p(k); 
                MSE_best = curMSE; 
                MSE_all = sum(MSE) ./ (task_end - task_start + 1)'; 
                R2 = curR2; 
            end
            
            fprintf('gamma = %g, lambda = %g, p = %g, mean_MSE = %g, mean_R2 = %g (%g, %g, %g, %g, %g)\n', ...
                log2(gamma(i)), log2(lambda(j)), log2(p(k)), sum(sum(MSE))/numel(trnY), mean(curR2), ...
                log2(gamma_best), log2(lambda_best), log2(p_best), MSE_best, mean(R2)); 
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random permutation by swapping i and j instance for each class
function [svm_inst, svm_lbl] = random_perm(svm_inst, svm_lbl)
n = numel(svm_lbl);
rand('state', 0);
for i = 1: n
    k = round(i + (n - i)*rand());   % [i, n]
    svm_inst([k, i], :) = svm_inst([i, k], :);
    svm_lbl([k, i]) = svm_lbl([i, k]); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [train_inst, train_lbl, test_inst, test_lbl] = folding(svm_inst, svm_lbl, fold, k)
n= numel(svm_lbl); 

% folding instances
start_index = round((k - 1)*n/fold) + 1;
end_index = round(k*n/fold);
test_index = start_index: end_index;

% extract test instances and corresponding labels
test_inst = svm_inst(test_index, :);
test_lbl = svm_lbl(test_index);

% extract train instances and corresponding labels
train_inst = svm_inst;
train_inst(test_index, :) = [];
train_lbl = svm_lbl;
train_lbl(test_index) = [];
