% Step 1: Load the trained ResNet-18 model
load('trainedResNet18.mat', 'trainedNet'); % Make sure this matches the saved model name

% Step 2: Define the path to the folder containing subfolders with images
datasetsFolder = '/Users/stephanos/Desktop/training/datasets';

% Step 3: Initialize a results structure to store classifications
results = struct();

% Step 4: Loop through each folder and classify images
folders = dir(datasetsFolder);
folders = folders([folders.isdir] & ~ismember({folders.name}, {'.', '..'})); % Ignore '.' and '..'

for i = 1:length(folders)
    folderPath = fullfile(datasetsFolder, folders(i).name);
    images = dir(fullfile(folderPath, '*.jpg')); % Change '*.jpg' if your images are in another format
    
    % Create a structure for each folder's results
    folderResults = struct();
    
    for j = 1:length(images)
        imagePath = fullfile(folderPath, images(j).name);
        I = imread(imagePath);
        
        % Resize image to 224x224 and ensure RGB format for ResNet-18
        R = imresize(I, [224, 224]);
        if size(R, 3) == 1
            R = cat(3, R, R, R); % Convert grayscale to RGB if needed
        end
        
        % Classify the image and get the predicted label and probability
        [label, scores] = classify(trainedNet, R);
        probability = max(scores); % Get the highest probability score
        
        % Store the classification results
        folderResults(j).imageName = images(j).name;
        folderResults(j).label = label;
        folderResults(j).probability = probability;
        
        % Display the result in the Command Window
        fprintf('Image: %s, Label: %s, Probability: %.2f\n', images(j).name, string(label), probability);
    end
    
    % Save each folder's results in the main results structure
    results.(folders(i).name) = folderResults;
end

% Step 5: Save the entire results structure to a .mat file
save('resnet18_classification_results.mat', 'results');
disp('Classification results saved to resnet18_classification_results.mat');
