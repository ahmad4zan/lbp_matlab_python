function [accuracy, precision, recall, f1Score] = calculateMetrics(confMat)
    %% confmat
    % Extract TP, TN, FP, FN from the confusion matrix
    numClasses = size(confMat, 1);
     TPs =0; TNs =0; FPs=0; FNs =0;
for k = 1:numClasses
    TP = confMat(k, k);
    FP = sum(confMat(:, k)) - TP;
    FN = sum(confMat(k, :)) - TP;
    TN = sum(confMat(:)) - (TP + FP + FN);
    
    % Display the results
    fprintf('For class %d:\n', k);
    fprintf('TP: %d\n', TP);
    fprintf('FP: %d\n', FP);
    fprintf('FN: %d\n', FN);
    fprintf('TN: %d\n', TN);

    TPs = TPs + TP;
    FPs = FPs + FP;
    FNs = FNs + FN;
    TNs = TNs + TN;
end    
    % Calculate metrics
    accuracy  = (TPs + TNs) / (TPs + TNs + FPs + FNs);
    precision = TPs / (TPs + FPs);
    recall    = TPs / (TPs + FNs);
    f1Score   = 2*(precision*recall)/(precision + recall);
end