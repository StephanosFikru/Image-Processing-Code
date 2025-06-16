% Load ResNet-18 model
net = resnet18;

% Set up image datastore with training and validation datasets
Dataset = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset, 0.7);

% Define the input size of ResNet-18
Input_Layer_Size = net.Layers(1).InputSize(1:2);

% Resize training and validation images to match ResNet-18 input
Resized_Training_Image = augmentedImageDatastore(Input_Layer_Size, Training_Dataset, 'ColorPreprocessing', 'gray2rgb');
Resized_Validation_Image = augmentedImageDatastore(Input_Layer_Size, Validation_Dataset, 'ColorPreprocessing', 'gray2rgb');

% Modify ResNet-18 to match the number of classes in your dataset
Feature_Learner = net.Layers(69); % Last fully connected layer in ResNet-18
Output_Classifier = net.Layers(71); % Classification layer in ResNet-18
Number_of_Classes = numel(categories(Training_Dataset.Labels));

% Replace the last layers with new layers suitable for the number of classes in your data
New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
New_Classifier_Layer = classificationLayer('Name', 'new_classoutput');

% Create a layer graph and replace the layers
Layer_Graph = layerGraph(net);
Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

% Optional: Analyze the modified network
analyzeNetwork(Layer_Graph);

% Training options
Size_of_Minibatch = 5;
Validation_Frequency = floor(numel(Resized_Training_Image.Files) / Size_of_Minibatch);
Training_Options = trainingOptions('sgdm', ...
    'MiniBatchSize', Size_of_Minibatch, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the modified ResNet-18 network
trainedNet = trainNetwork(Resized_Training_Image, Layer_Graph, Training_Options);

% Save the trained network
save('trainedResNet18.mat', 'trainedNet');