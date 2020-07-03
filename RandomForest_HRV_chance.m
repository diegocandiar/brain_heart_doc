function [acc_rand] = RandomForest_HRV_chance(data, numrand, train1, train2, test1, test2)
% This function performs a Random Forest classification using arbitrary
% patients label. This function performs several classifications to obtain a
% chance accuracy distribution for unbalanced datasets. This function uses
% Random Forest console as implemented in: https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% data: struct with all patients data. data must contain the following
% subfields
% data.ibi: cell array with interbeat intervals
% data.time: cell array with time arrays of original ECG
% data.indx: cell array with R peaks samples respect data.time
% numrand: number of classifications to perform
% train1: number of patients considered as class1 in training set
% train2: number of patients considered as class2 in training set
% test1: number of patients considered as class1 in test set
% test2: number of patients considered as class2 in test set
% Outputs:
% acc_rand: chance accuracy distribution
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

%% compile Random Forest

compile_windows 

%% FEATURES TO USE
%define frequencies: these values correspond to typical frequency bands
%limits used in HRV plus 3 equally spaced values for each band using 4
%decimal digits
freq = [0.0001 0.01 0.02 0.03 0.04 0.0675 0.095 0.1225 0.15 0.2125 0.275 0.3375 0.4];

features_names = {};
for i = 1:length(freq)
    features_names{i} = sprintf('%4.3f Hz',freq(i));
end

%%
L = length(data.ibi);
fs = 2; %sample frequency for resampling IBI time series
nfft = 1024; % spectrum resolution
order = 7; % Burg order, the one used in the article is 7

acc_rand = zeros(1,numrand);

for n_rand= 1:numrand
    %% random sort
    patients = randperm(L);
    
    % random patients distribution
    set1_train = patients(1 : train1);
    set2_train = patients(train1+1 : train1+train2);   
    set1_test = patients(train1+train2+1 : train1+train2+test1);
    set2_test = patients(train1+train2+test1+1 : train1+train2+test1+test2);
    
    TEST = [set1_test set2_test];
    Y_tst = [ones(1,test1) zeros(1,test2)];

    data_set1 = zeros(train1, length(freq));
    for i = 1:train1
        % read data
        RR = data.ibi{set1_train(i)};
        time = data.time{set1_train(i)};
        indx = data.indx{set1_train(i)};
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

    data_set2 = zeros(train2, length(freq));
    for i = 1:train2
        % read data
        RR = data.ibi{set2_train(i)};
        time = data.time{set2_train(i)};
        indx = data.indx{set2_train(i)};
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

    data_set3 = zeros(length(TEST), length(freq));
    for i = 1:length(TEST)
        % read data
        RR = data.ibi{TEST(i)};
        time = data.time{TEST(i)};
        indx = data.indx{TEST(i)};
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
    Y_trn = [ones(train1,1); zeros(train2,1)];
    X_tst = [data_set3(:,feat)];

    model = classRF_train(X_trn,Y_trn, trees, nfeat);

    %% TESTING
    [Y_hat, votes] = classRF_predict(X_tst,model);
    
    acc_rand(n_rand) = length(find(Y_hat' == Y_tst))/length(Y_hat);
end

