clear,clc
load('net_2d.mat');

fullSet = imageDatastore(".../raw/"); % raw
fullRes = imageDatastore(".../gt/"); % GT

[imdsTrain, imdsTest, imdsVal,...
    pxdsTrain, pxdsTest, pxdsVal] = partitiondataIMAGES(fullSet,fullRes,0.85);

dsTrain = combine(imdsTrain,pxdsTrain);
dsVal   = combine(imdsVal,pxdsVal);
dsTest  = combine(imdsTest,pxdsTest);

inputSize = [size(imread(imdsTrain.Files{1})),1];
convlayers=setNetwork(inputSize,4,8);
options = trainingOptions('adam', ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',25, ...
        'L2Regularization',0.1,...
        'MaxEpochs',200, ...
        'MiniBatchSize',50, ...
        'InitialLearnRate',1e-1, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'ExecutionEnvironment','parallel',...
        'ValidationData',dsVal,...
        'ValidationFrequency',150,...
        'Verbose',true);


net_2d_new = trainNetwork(dsTrain,layerGraph(net_2d) ,options);
save('net_2d_new.mat','net_2d_new');

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = ...
    partitiondataIMAGES(imds,pxds,perc)
% Partition data by randomly selecting 60% of the data for training. The
% rest is used for testing.
% 通过随机选择60%的数据进行训练来对数据进行分区。其余部分用于测试。
    
% Set initial random state for example reproducibility.
% 设置初始随机状态
% rng(0); 
numFiles = numel(imds.Files);%像素数
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(perc(1)* numFiles);
% Use 50% of the images for validation and test
M = round((numFiles-N)/2)+1;

trainingIdx   = shuffledIndices(1:N);
validationIdx = shuffledIndices(N+1:(N+M));%验证

% Use the rest for testing.
testIdx = shuffledIndices((N+M)+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
valImages = imds.Files(validationIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);
imdsVal = imageDatastore(valImages);


% Create pixel label datastores for training and test.
% 创建用于训练和测试的像素标签数据存储。
trainingLabels  = pxds.Files(trainingIdx);
testLabels      = pxds.Files(testIdx);
valLabels       = pxds.Files(validationIdx);
pxdsTrain       = imageDatastore(trainingLabels);
pxdsTest        = imageDatastore(testLabels);
pxdsVal         = imageDatastore(valLabels);
end


function lgraph=setNetwork(imsize,nr_Layers,nr_FirstFilters)
%
% SETNETWORK generates the layer graph for an upU-Net architecture.
% SETNETWORK为upU-Net架构生成层图。
% graph=setNetwork(imsize,nr_Layers,nr_FirstFilters)
%
% MANDATORY INPUT 必填输入
% 
% imsize (integer array)    : 2D array with the size of the input images.具有输入图像大小的 2D 数组。

%
% nr_Layers (integer)       : number of blocks of layers in the contractive
%                             and expansive parts.收缩路径和扩展路径中的层块数。
%
% nr_FirstFilters (integer) : number of filters in the very first block.
%                             This number will doubled in the subsequent
%                             blocks.第一个block中的过滤器数量。这个数字将在随后的区块中翻倍。
%
% OUTPUT
%
% lgraph (layer graph)      : layer graph to be trained.要训练的层图。
%
%==========================================================================
%
% Input layer
convlayers = imageInputLayer(imsize,'Name','input',...
    'Normalization','zscore');


% Contractive path 收缩路径
for i = 1:nr_Layers
    convlayers = [convlayers
        
    convolution2dLayer(3,nr_FirstFilters,...
    'Name',sprintf('Conv%d',i),...
    'Stride',[2,2],...
    'Padding',[1 1],...
    'PaddingValue','symmetric-exclude-edge');

    batchNormalizationLayer;
    
    reluLayer('Name',sprintf('relu%d',i ))

    ];
    nr_FirstFilters= nr_FirstFilters*2;
end

% Expansive path 扩展路径
% 
for i = 1:nr_Layers
    nr_FirstFilters= nr_FirstFilters/2;
    convlayers = [convlayers

    transposedConv2dLayer(2,nr_FirstFilters,...
    'Name',sprintf('TransConv%d',i),...
    'Stride',2)
    reluLayer('Name',sprintf('relu%d',nr_Layers+i))
    additionLayer(2,'Name',sprintf('add%d',nr_Layers+i));

    ];

end

% Final layers and Regression layer for training 用于训练的最终层和回归层
convlayers = [convlayers,
    convolution2dLayer(1,1,'Name','1d1');
    reluLayer('Name','relufin');
    regressionLayer('Name','RegressionOutput')];


% Connecting the contractive path and the expansive path 连接收缩路径和扩展路径
lgraph = layerGraph(convlayers);
for i=1:nr_Layers
    skipConv = transposedConv2dLayer(2,nr_FirstFilters,...
        'Stride',2,...
        'Name',sprintf('skipConv%d',i));
    nr_FirstFilters= nr_FirstFilters*2;
    lgraph = addLayers(lgraph,skipConv);
end


for i = 1:nr_Layers
    lgraph = connectLayers(lgraph,sprintf('relu%d',i),...
                                  sprintf('skipConv%d',i));
    lgraph = connectLayers(lgraph,sprintf('skipConv%d',i),...
        sprintf('add%d/in2',2*nr_Layers-i+1));
end
end


