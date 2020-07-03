function [Acc, Y_hat, Score, Gini] = RandomForest_HRV_validation(data1, data2, data3, Y_tst)
% This function performs a Random Forest validation classification. This
% function uses Random Forest console as implemented in:
% https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% data1 data2: struct with training data of class1 and class2. 
% data3: struct with test data. 
% data structs must contain the following
% subfields
% data.ibi: cell array with interbeat intervals
% data.time: cell array with time arrays of original ECG
% data.indx: cell array with R peaks samples respect data.time
% Y_test: logical array with test set class. 1 = class1, 0= class2
% 
% Outputs:
% Acc: validation accuracy
% Y_hat: logical array with test patients predicted classes.
% Gini: importance index for each feaure (1 x feature)
% features_names: cell array with features names
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

%% compile Random Forest

compile_windows 

%% read
L1 = length(data1.ibi);
L2 = length(data2.ibi);
L3 = length(data3.ibi);

%% FEATURES TO USE
%define frequencies: these values correspond to typical frequency bands
%limits used in HRV plus 3 equally spaced values for each band using 4
%decimal digits
freq = [0.0001 0.01 0.02 0.03 0.04 0.0675 0.095 0.1225 0.15 0.2125 0.275 0.3375 0.4];

features_names = {};
for i = 1:length(freq)
    features_names{i} = sprintf('%4.3f Hz',freq(i));
end

%% COMPUTE FEATURES
fs = 2; %sample frequency for resampling IBI time series
nfft = 1024; % spectrum resolution
order = 7; % Burg order, the one used in the article is 7

data_set1 = zeros(L1, length(freq));
for i = 1:L1
    % read data
    RR = data1.ibi{i};
    time = data1.time{i};
    indx = data1.indx{i};
    % create IBI time series
    t = time(indx(1:length(RR)))';
    y=RR.*1000; %convert ibi to ms  
    t2 = t(1):1/fs:t(length(t)); %time values 
    y2=interp1(t,y,t2,'spline'); %cubic spline interpolation
    %compute spectrum
    [PSD,F]=pburg(y2,order,(nfft*2)-1,fs,'onesided');
    %store
    data_set1(i,:) = interp1(F,PSD,freq); 
end

data_set2 = zeros(L2, length(freq));
for i = 1:L2
    % read data
    RR = data2.ibi{i};
    time = data2.time{i};
    indx = data2.indx{i};
    % create IBI time series
    t = time(indx(1:length(RR)))';
    y=RR.*1000; %convert ibi to ms  
    t2 = t(1):1/fs:t(length(t)); %time values 
    y2=interp1(t,y,t2,'spline'); %cubic spline interpolation
    %compute spectrum
    [PSD,F]=pburg(y2,order,(nfft*2)-1,fs,'onesided');
    %store
    data_set2(i,:) = interp1(F,PSD,freq); 
end

data_set3 = zeros(L3, length(freq));
for i = 1:L3
    % read data
    RR = data3.ibi{i};
    time = data3.time{i};
    indx = data3.indx{i};
    % create IBI time series
    t = time(indx(1:length(RR)))';
    y=RR.*1000; %convert ibi to ms  
    t2 = t(1):1/fs:t(length(t)); %time values 
    y2=interp1(t,y,t2,'spline'); %cubic spline interpolation
    %compute spectrum
    [PSD,F]=pburg(y2,order,(nfft*2)-1,fs,'onesided');
    %store
    data_set3(i,:) = interp1(F,PSD,freq); 
end

%% TRAINING
feat = [1:length(freq)];
nfeat = floor(length(feat)^0.5);
trees = 1000;

X_trn = [data_set1(:,feat); data_set2(:,feat)];
Y_trn = [ones(L1,1); zeros(L2,1)];
X_tst = [data_set3(:,feat)];

clear extra_options
extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)
model = classRF_train(X_trn,Y_trn, trees, nfeat, extra_options);

[Y_hat, votes] = classRF_predict(X_tst,model);

Acc = length(find(Y_hat == Y_tst))/length(Y_hat);

%% Compute consciousness scores

Score = zeros(1, L3);
for i = 1:length(Y_hat)
    Score(i) = votes(i,2)/trees;
end

%% Get feature importance

Gini = model.importance(:,end);

figure
bar(Gini, 'FaceColor',[0 0.75 .25]); 
title('Feature importance - Gini index'); ylabel('magnitude');
xlim([0 length(feat)+1])
set(gcf,'units','points','position',[10,10,1200,400])
xticklabel_rotate(1:length(feat),45,[features_names(1,feat)])

end

