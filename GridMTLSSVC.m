function [gamma_best, lambda_best, p_best, precision_best, precision_all] = GridMTLSSVC(trnX, trnY, trnN, fold, gamma_best, lambda_best, p_best, precision_best)
% 
% [gamma_best, lambda_best, p_best, precision_best, precision_all] =
% GridMTLSSVC(trnX, trnY, task_start, fold, gamma_best, lambda_best, p_best,
% precision_best); 
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

precision = zeros(fold, T); 
for i = 1: length(gamma) 
    for j = 1: length(lambda)
        for k = 1: length(p) 
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
                [alpha, b] = MTLSSVCTrain(train_inst, train_lbl, trnN, gamma(i), lambda(j), p(k)); 
                [predictY, confusionMatrix]= MTLSSVCPredict(test_inst, test_lbl, tstN, train_inst, train_lbl, trnN, alpha, b, lambda(j), p(k)); 
                precision(v, :) = sum(confusionMatrix(:, 1: 2), 2)'; 
            end
            
            if precision_best < (sum(sum(precision)) / numel(trnY))
                gamma_best = gamma(i); 
                lambda_best = lambda(j); 
                p_best = p(k); 
                precision_best = (sum(sum(precision)) / numel(trnY));
                precision_all = sum(precision)' ./ (task_end - task_start + 1); 
            end
            
            fprintf('gamma = %g, lambda = %g, p = %g, mean_precision = %g (%g, %g, %g, %g)\n', ...
                log2(gamma(i)), log2(lambda(j)), log2(p(k)), sum(sum(precision)) / numel(trnY), ...
                log2(gamma_best), log2(lambda_best), log2(p_best), precision_best); 
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random permutation by swapping i and j instance for each class
function [svm_inst, svm_lbl] = random_perm(svm_inst, svm_lbl)
index{1} = find(svm_lbl == +1);
index{2} = find(svm_lbl == -1);
l = [numel(index{1}), numel(index{2})];
rand('state', 0);
for i = 1: 2
    for j = 1: l(i)
        k = round(j + (l(i) - j)*rand());   % [j, l_plus] or [j, l_minus]
        svm_inst([index{i}(k), index{i}(j)], :) = svm_inst([index{i}(j), index{i}(k)], :);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [train_inst, train_lbl, test_inst, test_lbl] = folding(svm_inst, svm_lbl, fold, k)
positive_index = find(svm_lbl == +1);
negative_index = find(svm_lbl == -1);
l_plus = numel(positive_index);
l_minus = numel(negative_index);

% folding positive instances
start_index = round((k - 1)*l_plus/fold) + 1;
end_index = round(k*l_plus/fold);
test_index = positive_index(start_index: end_index);

% folding negative instances
start_index = round((k - 1)*l_minus/fold) + 1;
end_index = round(k*l_minus/fold);
test_index = [test_index; negative_index(start_index: end_index)];

% extract test instances and corresponding labels
test_index = sort(test_index, 'descend');
test_inst = svm_inst(test_index, :);
test_lbl = svm_lbl(test_index);

% extract train instances and corresponding labels
train_inst = svm_inst;
train_inst(test_index, :) = [];
train_lbl = svm_lbl;
train_lbl(test_index) = [];
