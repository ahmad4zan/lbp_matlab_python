clc;    % Clear the command window.
close all;  % Close all figures.
clear all;  % Clear all variables.

LbpType = 'riu2'; % Options: 'ri', 'riu2', 'u2'
desiredDimensions = 10; % Set the desired number of dimensions
trainRatio = 0.8; % Percentage of training images

%% Inserting the malimg dataset
imageFolder = 'malimg_dataset'; %'malimg_dataset/test';
categories = dir(imageFolder);
categories = categories(~ismember({categories.name}, {'.', '..'})); % Remove '.' and '..' from the list

numCategories = numel(categories);
images = {};
labels = categorical([]);

for i = 1:numCategories
    categoryPath = fullfile(imageFolder, categories(i).name);
    imageFiles = dir(fullfile(categoryPath, '*.png')); % Assuming images are in PNG format
    
    for j = 1:numel(imageFiles)
        imagePath = fullfile(categoryPath, imageFiles(j).name);
        image = imread(imagePath);
        images{end+1} = image;
        labels(end+1) = categories(i).name;
    end
end

% Initialize variables for storing features and labels
numImages = numel(images);
featureMatrix = [];

mapping = getmapping(8, LbpType); % Create mapping u2, ri, riu2

 %% Visualize before LBP extraction
% % Extract mean and standard deviation of pixel values
% meanValues = zeros(numImages, 1);
% stdValues = zeros(numImages, 1);
% 
% for i = 1:numImages
%     imageDouble = double(images{i}); % Convert image to double
%     meanValues(i) = mean(imageDouble(:));
%     stdValues(i) = std(imageDouble(:));
% end
% 
% % Plot the mean and standard deviation
% figure;
% gscatter(meanValues, stdValues, labels);
% xlabel('Mean Pixel Value');
% ylabel('Standard Deviation of Pixel Values');
% title('Scatter Plot of Images Before LBP Extraction');
% hold off;
% 
%% Feature extraction using LBP
for i = 1:numImages
    featureVector = LBP(images{i}, 1, 8, mapping, 'h'); % LBP histogram in (8,1) neighborhood
    featureMatrix(i, :) = featureVector;
end
% 
% %% Visualize after LBP extraction1
% % Extract mean and standard deviation of pixel values
% meanValues1 = zeros(numImages, 1);
% stdValues1 = zeros(numImages, 1);
% 
% for i = 1:numImages
%     imageDouble1 = double(featureMatrix(i,:)); % Convert image to double
%     meanValues1(i) = mean(imageDouble1(:));
%     stdValues1(i) = std(imageDouble1(:));
% end
% 
% % Plot the mean and standard deviation
% figure;
% gscatter(meanValues1, stdValues1, labels);
% xlabel('Mean Pixel Value');
% ylabel('Standard Deviation of Pixel Values');
% title('Scatter Plot of Images After LBP Extraction');
% 
% %% New projection feature, reature reduction
% % p = 10;    
% [~, pca_scores, ~, ~, var_explained] = pca(featureMatrix, 'NumComponents', desiredDimensions);
% 
% %% Visualize after LBP extraction2
% % Plot the first two dimensions of the LBP feature vectors
% for i = 1:numImages
%     imageDouble2 = double(pca_scores(i,:)); % Convert image to double
%     meanValues2(i) = mean(imageDouble2(:));
%     stdValues2(i) = std(imageDouble2(:));
% end
% 
% figure;
% % gscatter(pca_scores(:, 1), pca_scores(:, 2), labels);
% gscatter(meanValues2, stdValues2, labels);
% xlabel('Mean Pixel Value');
% ylabel('Standard Deviation of Pixel Values');
% title('Scatter Plot of Images After LBP Extraction & PCA Feature Reduction');
% hold off;

%% Splitting data

rng('default'); % For reproducibility

% Count the number of unique labels (types of malware)

uniqueLabels = unique(labels);
numLabels = numel(uniqueLabels);

% numTrainTotal = round(trainRatio * numImages);
% numTestTotal = numImages - numTrainTotal;

% Initialize variables for training and testing sets
trainFeatures = [];
trainLabels = categorical([]);
testFeatures = [];
testLabels = categorical([]);

%% Split the data into training and testing sets for each label

for i = 1:numLabels% Split the data into training and testing sets for each label
    % Get the indices of the current label
    labelIndices = find(labels == uniqueLabels(i));
    % if  i==1
    
        % Shuffle the indices randomly
        shuffledIndices = randperm(numel(labelIndices));

        % Calculate the number of training and testing samples
        numTrain = round(trainRatio * numel(labelIndices));
        numTest = numel(labelIndices) - numTrain;
    % end
    
    % Select the indices for training and testing
    trainIndices = labelIndices(shuffledIndices(1:numTrain));
    testIndices = labelIndices(shuffledIndices(numTrain+1:end));
    
    % Add the features and labels to the respective sets
    trainFeatures = [trainFeatures; featureMatrix(trainIndices, :)];
    trainLabels = [trainLabels, labels(trainIndices)];
    testFeatures = [testFeatures; featureMatrix(testIndices, :)];
    testLabels = [testLabels, labels(testIndices)];
end

%% Split the data into training and testing sets
%{
l = 1; m = 100
for k = 1:numCategories %assume 23
[trainInd, ~, testInd] = dividerand(l:m, trainRatio, 0, 1 - trainRatio);
trainFeatures = [trainFeatures; featureMatrix(trainInd, :)];
trainLabels = [trainLabels; labels(trainInd)];
testFeatures = [testFeatures; featureMatrix(testInd, :)];
testLabels = [testLabels; labels(testInd)];
l = l + 100
m = m + 100
end
% Ensure trainLabels is categorical
%testLabels = reshape(testLabels, 1, []);
%trainLabels = reshape(trainLabels, 1, []);
%}
% Initialize the feature matrix
% testLabelsfm = [];
% trainLabelsfm = [];
% 
% % Convert rows to a single row
% for i = 1:size(testLabels, 1)
%     testLabelsfm = [testLabelsfm, testLabels(i, :)];
% end
% for i = 1:size(trainLabels, 1)
%     trainLabelsfm = [trainLabelsfm, trainLabels(i, :)];
% end
% testLabels = testLabelsfm;
% trainLabels = trainLabelsfm;
trainLabels = categorical(trainLabels);

%% Splitting data
%{
rng('default'); % For reproducibility

% Split the data into training and testing sets
[trainInd, ~, testInd] = dividerand(numImages, trainRatio, 0, 1 - trainRatio);
trainFeatures = featureMatrix(trainInd, :);
trainLabels = labels(trainInd);
testFeatures = featureMatrix(testInd, :);
testLabels = labels(testInd);

% Ensure trainLabels is categorical
trainLabels = categorical(trainLabels);
%}


%% GMM Fitting

X = trainFeatures;
% numGMMs = numCategories; % Assuming 23 categories
GMModels = cell(1, numCategories);
numSamplesPerGMM = size(trainFeatures, 1) / numCategories;
regularizationValue = 0.01; % Regularization value to prevent ill-conditioned covariance

% % Plot the data points before creating GMMs
% figure;
% gscatter(X(:, 3), X(:, 5), trainLabels);
% xlabel('Feature 1');
% ylabel('Feature 2');
% title('Scatter Plot of Training Data Points');
% legend('Location', 'best');
% hold off;

for idx = 1:numCategories
    startIdx = (idx - 1) * numSamplesPerGMM + 1;
    endIdx = idx * numSamplesPerGMM;
    GMModels{idx} = fitgmdist(X(startIdx:endIdx, 1:10), 2, 'RegularizationValue', regularizationValue);
end

%% Create 2D GMMs for plotting

GMModels2D = cell(1, numCategories);
for idx = 1:numCategories
    startIdx = (idx - 1) * numSamplesPerGMM + 1;
    endIdx = idx * numSamplesPerGMM;
    GMModels2D{idx} = fitgmdist(X(startIdx:endIdx, 1:2), 1, 'RegularizationValue', regularizationValue);
end

%% Plotting without fcontour
figure;
gscatter(X(:, 1), X(:, 2), trainLabels);
hold on;

% Add labels and title
xlabel('Feature 1');
ylabel('Feature 2');
title('Gscatter Plot without Gaussian Mixture Model Contour Lines');
hold off;

%% Plotting with fcontour
figure;
%coeff = pca(X);
gscatter(X(:, 1), X(:, 2), trainLabels);
hold on;

% Loop through each 2D GMM and plot the contour lines using fcontour
for idx = 1:numCategories
    % Define the plotting range for this category
    categoryData = X((idx - 1) * numSamplesPerGMM + 1 : idx * numSamplesPerGMM, 1:2);
    xMin = min(categoryData(:, 1));
    xMax = max(categoryData(:, 1));
    yMin = min(categoryData(:, 2));
    yMax = max(categoryData(:, 2));
    xRange = linspace(xMin - 1, xMax + 1, 100);
    yRange = linspace(yMin - 1, yMax + 1, 100);
    [XGrid, YGrid] = meshgrid(xRange, yRange);
    
    % Plot the contour lines for this category
    gmPDF = @(x, y) arrayfun(@(x0, y0) pdf(GMModels{idx}, [x0 y0]), x, y);
    fcontour(gmPDF, [min(xRange) max(xRange) min(yRange) max(yRange)], 'LineColor', rand(1,3), 'HandleVisibility', 'off');
    
end

% Add labels and title
xlabel('Feature 1');
ylabel('Feature 2');
title('Gscatter Plot with Gaussian Mixture Model Contour Lines');

% Set the legend only for the scatter plot
legend(categories.name, 'Location', 'Best');
hold off;

%% Predicting Labels for Test Data using posterior
numTestImages = size(testFeatures, 1);
predictedLabels = strings(numTestImages, 1);

% Loop through each test image
for i = 1:numTestImages
    logLikelihoods = zeros(numCategories, 1);
    % Calculate the negative log-likelihood for each GMM
    for j = 1:numCategories
        [~, nlogL] = posterior(GMModels{j}, testFeatures(i, 1:10));
        logLikelihoods(j) = -nlogL; % Use negative log-likelihood
    end
    % Assign the category with the highest log-likelihood
    [~, maxIdx] = max(logLikelihoods);
    predictedLabels(i) = categories(maxIdx).name;
end

predictedLabels = categorical(predictedLabels);

%% Evaluation Metrics
confMat = confusionmat(testLabels, predictedLabels);
accuracy = sum(diag(confMat)) / sum(confMat, 'all');
[accuracy, precision, recall, f1Score] = calculateMetrics(confMat);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n',recall * 100);
fprintf('F1Score: %.2f%%\n', f1Score * 100);

%% Confusion Matrix Plot
figure;
confusionchart(confMat, {categories.name});
title('Confusion Matrix');

%% Visualizing Test Data
figure;
gscatter(testFeatures(:, 1), testFeatures(:, 2), predictedLabels);
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter Plot of Test Data with Predicted Labels');
legend('Location', 'best');
hold off;

%% Initialize arrays to store accuracies for each GMM
accuracies = zeros(1, numCategories);

% Loop through each GMM model
for gmmIdx = 1:numCategories
   
%  predictedLabels = strings(numTestImages, 1);
% 
%     % Predict the label for each test image using the current GMM
%     for i = 1:numTestImages
%         % Compute the posterior probability (negative log-likelihood) for the current test sample
%         [~, nlogL] = posterior(GMModels{gmmIdx}, testFeatures(i, 1:10));
%         logLikelihood = -nlogL; % Use negative log-likelihood
% 
%         % Assign the category based on the GMM with the highest log-likelihood
%         predictedLabels(i) = categories(gmmIdx).name;
%     end
% 
%     % Convert the predicted labels to categorical
%     predictedLabels = categorical(predictedLabels);
% 
%     % Compute the confusion matrix for the current GMM
%     confMat = confusionmat(testLabels, predictedLabels);

    % Calculate accuracy for the current GMM
    a = diag(confMat);
    accuracies(gmmIdx) = a(gmmIdx) / 20;

    % Display the accuracy for the current GMM
    fprintf('Accuracy for GMM %d: %.2f%%\n', gmmIdx, accuracies(gmmIdx) * 100);
end

% Plot the accuracies of each GMM
figure;
bar(1:numCategories, accuracies * 100);
xlabel('GMM Index');
ylabel('Accuracy (%)');
title('Accuracy of Each GMM on Test Data');
