%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code trains an SVM classifier
% 
% Input:
%    trainFolder --> Location of the training image set
%     testFolder --> Location of the test image set
%   signsTrained --> Label of the signs for which training is to be done
%       gridSize --> Grdi size to be used for HOG feature extraction
%    codingStyle --> Coding to bbe used when training the SVM model
% 
% Output:
%   model --> Trained model
%   score --> Score matrix for the trained model
% 
% Submitted by: Ashwin Goyal (UID - 115526297)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [model, score] = trainSVMclassifier(trainFolder, testFolder, signsTrained, gridSize, codingStyle)

    % Get training folders
    trainDir = dir(trainFolder);
    trainDir(strcmp({trainDir.name},'.') | strcmp({trainDir.name},'..') | ([trainDir.isdir] == 0)) = [];
    
    % Get HOG features for training data
    disp('Reading training data')
    trainX = []; trainY = [];
    for i = 1:length(trainDir)
        currInfo = readtable([trainFolder trainDir(i).name '\GT-' trainDir(i).name '.csv']);
        for j = 1:size(currInfo,1)
            if any(signsTrained == currInfo.ClassId(j))
                I = imread([trainFolder trainDir(i).name '\' currInfo.Filename{j}]);
                I = imresize(I(currInfo.Roi_X1(j):currInfo.Roi_X2(j),currInfo.Roi_Y1(j):currInfo.Roi_Y2(j),:),[64 64]);
                features = extractHOGFeatures(I,'CellSize',gridSize);
                trainX = [trainX; features];
                trainY = [trainY; currInfo.ClassId(j)];
            end
        end
    end
    
    % Get test folders
    testDir = dir(testFolder);
    testDir(strcmp({testDir.name},'.') | strcmp({testDir.name},'..') | ([testDir.isdir] == 0)) = [];
    
    % Get HOG features for test data
    disp('Reading test data')
    testX = []; testY = [];
    for i = 1:length(testDir)
        currInfo = readtable([testFolder testDir(i).name '\GT-' testDir(i).name '.csv']);
        for j = 1:size(currInfo,1)
            if any(signsTrained == currInfo.ClassId(j))
                I = imread([testFolder testDir(i).name '\' currInfo.Filename{j}]);
                I = imresize(I(currInfo.Roi_X1(j):currInfo.Roi_X2(j),currInfo.Roi_Y1(j):currInfo.Roi_Y2(j),:),[64 64]);
                features = extractHOGFeatures(I,'CellSize',gridSize);
                testX = [testX; features];
                testY = [testY; currInfo.ClassId(j)];
            end
        end
    end
    
    % Train SVM classifier
    disp(['Training classifier with ' codingStyle ' Coding'])
    model = fitcecoc(trainX,trainY,'Coding',codingStyle,'Options',statset('UseParallel',true));
    
    % Use trained classifier to predict data
    disp('Testing the trained classifier')
    labels = predict(model,testX);
    
    % Get individual scores
    score = zeros(8);
    for i = 1:length(labels)
        score(signsTrained == testY(i),signsTrained == labels(i)) = score(signsTrained == testY(i),signsTrained == labels(i)) + 1;
    end
    
    % Check accuracy of prediction
    accuracy = sum(labels==testY)/length(testY);
    disp(['The SVM classifier has an accuracy of ' num2str(accuracy*100,'%.2f') '%. Thus, the model is acceptable.'])

end