%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code runs the entire traffic sign detection as well as
% classification algorithm
% 
% Submitted by: Ashwin Goyal (UID - 115526297)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the image folders
testFolder = '..\input\Testing\';
trainFolder = '..\input\Training\';
inputFolder = '..\input\input\';
labelsFolder = '..\input\labels\';

% Train the SVM model
gridSize = [8 9];
requiredSigns = [1 14 17 19 21 35 38 45];
coding = 'sparserandom';
[model,score] = trainSVMclassifier(trainFolder,testFolder,requiredSigns,gridSize,coding);

% Define some tuning parameters
minArea = 100;
maxAspectRatio = 1.5;
lastBlueSign = [];
lastRedSign = [];

% Create video object
vidObj = VideoWriter('..\output\trafficSignDetection.mp4','MPEG-4');
open(vidObj)

% Create figure
figure('units','normalized','outerposition',[0 0 1 1])

% Read images
inputImgs = dir([inputFolder '*.jpg']);
for i = 1:length(inputImgs)
    % Read the image
    I = im2double(imread([inputFolder inputImgs(i).name]));
    Ihsv = rgb2hsv(I);
    imshow(I)
    hold on
    
    % Remove noise
    Ifilt = imguidedfilter(I);
    Ifilt = im2double(locallapfilt(im2uint8(Ifilt),0.2,5,'NumIntensityLevels',16));
    
    % Normalize the image intensities
    Ired = min(cat(3,(Ifilt(:,:,1) - Ifilt(:,:,3)),(Ifilt(:,:,1) - Ifilt(:,:,2))),[],3)./sum(Ifilt,3);
    Ired = max(cat(3,zeros(size(Ifilt,1),size(Ifilt,2)),Ired),[],3);
    Iblue = min(cat(3,(Ifilt(:,:,3) - Ifilt(:,:,1)),(Ifilt(:,:,3) - Ifilt(:,:,2))),[],3)./sum(Ifilt,3);
    Iblue = max(cat(3,zeros(size(Ifilt,1),size(Ifilt,2)),Iblue),[],3);
    
    % Detect blue signs
    regions = detectMSERFeatures(Iblue,'RegionAreaRange',[100 14000],'ROI',[1 1 size(I,2) 0.6*size(I,1)]);
    Iblue = false(size(Iblue));
    for j = 1:length(regions)
        Iblue(regions.PixelList{j}(:,2),regions.PixelList{j}(:,1)) = true;
    end
    % Integrate with HSV color space
    blueMask = Ihsv(:,:,2)>=0.45 & Ihsv(:,:,2)<=0.8 & Ihsv(:,:,3)>=0.35 & Ihsv(:,:,3)<=1;
    Iblue = Iblue & blueMask;
    % Make bounding boxes
    regions = regionprops(Iblue,'Area','BoundingBox');
    blueBox = []; blueArea = [];
    if ~isempty([regions.Area]) && any([regions.Area] >= minArea)
        blueArea = [regions.Area]';
        [~,ind] = sort(blueArea,'descend');
        blueBox = reshape([regions(ind).BoundingBox],4,length(regions))';
        ind = (blueArea >= minArea) & (blueBox(:,3)./blueBox(:,4) <= maxAspectRatio);
        blueBox = blueBox(ind,:);
        blueArea = blueArea(ind);
        if ~isempty(blueArea)
            [blueBox,blueArea] = selectStrongestBbox(blueBox,blueArea,'OverlapThreshold',0);
        end
    end

    % Detect red signs
    regions = detectMSERFeatures(Ired,'RegionAreaRange',[100 14000],'ROI',[0.6*size(I,2) 1 0.4*size(I,2) 0.6*size(I,1)]);
    Ired = false(size(Ired));
    for j = 1:length(regions)
        Ired(regions.PixelList{j}(:,2),regions.PixelList{j}(:,1)) = true;
    end
    % Integrate with HSV color space
    redMask = Ihsv(:,:,2)>=0.5 & Ihsv(:,:,2)<=0.9 & Ihsv(:,:,3)>=0.65 & Ihsv(:,:,3)<=0.75;
    Ired = Ired & redMask;
    % Make bounding boxes
    regions = regionprops(Ired,'Area','BoundingBox');
    redBox = []; redArea = [];
    if ~isempty([regions.Area]) && any([regions.Area] >= minArea)
        redArea = [regions.Area]';
        [~,ind] = sort(redArea,'descend');
        redBox = reshape([regions(ind).BoundingBox],4,length(regions))';
        ind = (redArea > minArea) & (redBox(:,3)./redBox(:,4) <= maxAspectRatio);
        redBox = redBox(ind,:);
        redArea = redArea(ind);
        if ~isempty(redArea)
            [redBox,redArea] = selectStrongestBbox(redBox,redArea,'OverlapThreshold',0);
        end
    end
            
    % Classify blue signs
    count = 0;
    tempLastBlueSign = [];
    if ~isempty(lastBlueSign)
        usedIndex = false(size(lastBlueSign,1),1);
    end
    for j = 1:size(blueBox,1)
        Iblue = imresize(imcrop(I,blueBox(j,:)),[64 64]);
        features = extractHOGFeatures(Iblue,'CellSize',gridSize);
        [~,score] = predict(model,features);
        [maxScore,ind] = max(score(6:8));
        if maxScore < -0.15
            if ~isempty(lastBlueSign) && any(all(abs(lastBlueSign(:,1:2) - flip(blueBox(j,1:2) + blueBox(j,3:4)/2)) < 20,2))
                index = all(abs(lastBlueSign(:,1:2) - flip(blueBox(j,1:2) + blueBox(j,3:4)/2)) < 20,2);
                rectangle('Position', blueBox(j,:),'EdgeColor','b','LineWidth',2)
                Iblue = imresize(imread([labelsFolder num2str(lastBlueSign(find(index,1),3)) '.jpg']),[blueBox(j,4) blueBox(j,3)]);
                image([int64(blueBox(j,1)-blueBox(j,3)) int64(blueBox(j,1)-blueBox(j,3))],[int64(blueBox(j,2)) int64(blueBox(j,2))],Iblue);
                usedIndex(find(index,1)) = true;
            end
        else
            if count == 0
                tempLastBlueSign = [flip(blueBox(j,1:2) + blueBox(j,3:4)/2) requiredSigns(5+ind) 0];
            else
                tempLastBlueSign = [tempLastBlueSign; flip(blueBox(j,1:2) + blueBox(j,3:4)/2), requiredSigns(5+ind), 0];
            end
            rectangle('Position', blueBox(j,:),'EdgeColor','b','LineWidth',2)
            Iblue = imresize(imread([labelsFolder num2str(requiredSigns(5+ind)) '.jpg']),[blueBox(j,4) blueBox(j,3)]);
            image([int64(blueBox(j,1)-blueBox(j,3)) int64(blueBox(j,1)-blueBox(j,3))],[int64(blueBox(j,2)) int64(blueBox(j,2))],Iblue);
            count = count + 1;
        end
    end
    if ~isempty(tempLastBlueSign)
        lastBlueSign = tempLastBlueSign;
    elseif ~isempty(lastBlueSign)
        lastBlueSign(~usedIndex,4) = lastBlueSign(~usedIndex,4) + 1;
        lastBlueSign(usedIndex,4) = 0;
        lastBlueSign(lastBlueSign(:,4) > 10,:) = [];
    end

    % Classify red signs
    count = 0;
    tempLastRedSign = [];
    if ~isempty(lastRedSign)
        usedIndex = false(size(lastRedSign,1),1);
    end
    for j = 1:size(redBox,1)
        Ired = imresize(imcrop(I,redBox(j,:)),[64 64]);
        features = extractHOGFeatures(Ired,'CellSize',gridSize);
        [~,score] = predict(model,features);
        [maxScore,ind] = max(score(1:5));
        if maxScore < -0.15
            if ~isempty(lastRedSign) && any(all(abs(lastRedSign(:,1:2) - flip(redBox(j,1:2) + redBox(j,3:4)/2)) < 20,2))
                index = all(abs(lastRedSign(:,1:2) - flip(redBox(j,1:2) + redBox(j,3:4)/2)) < 20,2);
                rectangle('Position', redBox(j,:),'EdgeColor','r','LineWidth',2)
                Ired = imresize(imread([labelsFolder num2str(lastRedSign(find(index,1),3)) '.jpg']),[redBox(j,4) redBox(j,3)]);
                image([int64(redBox(j,1)-redBox(j,3)) int64(redBox(j,1)-redBox(j,3))],[int64(redBox(j,2)) int64(redBox(j,2))],Ired);
                usedIndex(find(index,1)) = true;
            end
        else
            if count == 0
                tempLastRedSign = [flip(redBox(j,1:2) + redBox(j,3:4)/2) requiredSigns(ind) 0];
            else
                tempLastRedSign = [tempLastRedSign; flip(redBox(j,1:2) + redBox(j,3:4)/2), requiredSigns(ind), 0];
            end
            rectangle('Position', redBox(j,:),'EdgeColor','r','LineWidth',2)
            Ired = imresize(imread([labelsFolder num2str(requiredSigns(ind)) '.jpg']),[redBox(j,4) redBox(j,3)]);
            image([int64(redBox(j,1)-redBox(j,3)) int64(redBox(j,1)-redBox(j,3))],[int64(redBox(j,2)) int64(redBox(j,2))],Ired);
            count = count + 1;
        end
    end
    if ~isempty(tempLastRedSign)
        lastRedSign = tempLastRedSign;
    elseif ~isempty(lastRedSign)
        lastRedSign(~usedIndex,4) = lastRedSign(~usedIndex,4) + 1;
        lastRedSign(usedIndex,4) = 0;
        lastRedSign(lastRedSign(:,4) > 10,:) = [];
    end
    
    % Write the image to video
    writeVideo(vidObj,getframe(gca))
end

% Close the video object
close(vidObj)