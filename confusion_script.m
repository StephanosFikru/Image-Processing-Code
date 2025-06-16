% Load classification results for ResNet-18 and GoogLeNet
load('resnet18A_results.mat', 'results');
load('googlenetA_results.mat', 'results');

% Specify valid class labels
validLabels = {'H2O2_50_1_200', 'H2O2_25_micro_conc_1_200_2', 'PAL_1_200_JPG', ...
               'PAL_2_5_1_200', 'TREATED_W_H2O2_100_micro'};

% Initialize arrays for true and predicted labels
trueLabelsResNet18 = {};
predictedLabelsResNet18 = {};
trueLabelsGoogLeNet = {};
predictedLabelsGoogLeNet = {};

% Extract labels from ResNet-18 results
fields = fieldnames(results);
for i = 1:numel(fields)
    folderName = fields{i};
    folderResults = results.(folderName);
    
    % Check if folderName is a valid label and proceed
    if ismember(folderName, validLabels)
        for j = 1:numel(folderResults)
            trueLabelsResNet18{end+1} = folderName;
            predictedLabelsResNet18{end+1} = char(folderResults(j).label);
        end
    end
end

% Display extracted ResNet-18 labels for verification
disp('ResNet-18 True Labels:');
disp(trueLabelsResNet18);
disp('ResNet-18 Predicted Labels:');
disp(predictedLabelsResNet18);

% Extract labels from GoogLeNet results
fields = fieldnames(results);
for i = 1:numel(fields)
    folderName = fields{i};
    folderResults = results.(folderName);
    
    % Check if folderName is a valid label and proceed
    if ismember(folderName, validLabels)
        for j = 1:numel(folderResults)
            trueLabelsGoogLeNet{end+1} = folderName;
            predictedLabelsGoogLeNet{end+1} = char(folderResults(j).label);
        end
    end
end

% Display extracted GoogLeNet labels for verification
disp('GoogLeNet True Labels:');
disp(trueLabelsGoogLeNet);
disp('GoogLeNet Predicted Labels:');
disp(predictedLabelsGoogLeNet);

% Convert cell arrays to categorical format for ResNet-18 and GoogLeNet
trueLabelsResNet18 = categorical(trueLabelsResNet18, validLabels);
predictedLabelsResNet18 = categorical(predictedLabelsResNet18, validLabels);
trueLabelsGoogLeNet = categorical(trueLabelsGoogLeNet, validLabels);
predictedLabelsGoogLeNet = categorical(predictedLabelsGoogLeNet, validLabels);

% Plot the confusion matrix for ResNet-18
figure;
confusionchart(trueLabelsResNet18, predictedLabelsResNet18, ...
    'Title', 'Confusion Matrix for ResNet-18', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

% Plot the confusion matrix for GoogLeNet
figure;
confusionchart(trueLabelsGoogLeNet, predictedLabelsGoogLeNet, ...
    'Title', 'Confusion Matrix for GoogLeNet', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

disp('Confusion matrices generated for both models based on saved results.');
