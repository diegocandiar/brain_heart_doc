function [Acc, acc_fold, gini, features_names] = RandomForest_HRV_crossvalidation(data1, data2, nfolds)
% This function performs a Random Forest crossvalidation classification.
% This function divides the set of patients in folds to perform
% classifications using different set of patients each time. This function
% uses Random Forest console as implemented in:
% https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% data1 and data2: struct with patients data of class1 and class2. 
% data structs must contain the following
% subfields
% data.ibi: cell array with interbeat intervals
% data.time: cell array with time arrays of original ECG
% data.indx: cell array with R peaks samples respect data.time
% nfolds: number of folds (integer)
%
% Outputs:
% Acc: crossvalidation accuracy
% acc_fold: accuracy for individual folds
% Gini: importance index for each feaure (fold x feature)
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

%% FEATURES TO USE
%define frequencies: these values correspond to typical frequency bands
%limits used in HRV plus 3 equally spaced values for each band using 4
%decimal digits
freq = [0.0001 0.01 0.02 0.03 0.04 0.0675 0.095 0.1225 0.15 0.2125 0.275 0.3375 0.4];

features_names = cell(1,length(freq));
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

%% Fold distribution
% set random location to patients in the folds
set1_sort = randperm(L1);
set2_sort = randperm(L2);

% define how many patients are in each fold
fold_details = zeros(2,nfolds);
fold_details(1,1:nfolds-1) = floor(L1/nfolds); fold_details(1,nfolds) = L1 - (nfolds-1)*floor(L1/nfolds);
fold_details(2,1:nfolds-1) = floor(L2/nfolds); fold_details(2,nfolds) = L2 - (nfolds-1)*floor(L2/nfolds);

% create test cells with patient id in each fold
m = cell(1,nfolds);
n = cell(1,nfolds);
% create train cells with patient id in each fold
mb = cell(1,nfolds);
nb = cell(1,nfolds);

% iterate to define patients in each fold
k1 = 1; k2 = 1;
for i = 1 : nfolds
    % set 1 folds
    n_subjects = fold_details(1,i);
    subjects_test = set1_sort(k1 : (k1 + n_subjects -1));
    subjects_train = setdiff(set1_sort, subjects_test);
    m{i} = set1_sort(subjects_test);
    mb{i} = set1_sort(subjects_train);
    k1 = k1 + n_subjects;
    
    % set 2 folds    
    n_subjects = fold_details(2,i);
    subjects_test = set2_sort(k2 : (k2 + n_subjects -1));
    subjects_train = setdiff(set2_sort, subjects_test);
    n{i} = set1_sort(subjects_test);
    nb{i} = set1_sort(subjects_train);
    k2 = k2 + n_subjects;
end

%% Crossvalidation
nfeat = floor(length(freq)^0.5); % number of features per node in Random Forest
trees = 1000; % total number of trees in Random Forests

acc_fold = zeros(1,nfolds);
gini = zeros(nfolds, length(freq));
for j = 1:nfolds
    X_tst = []; Y_tst = [];
    for p = 1 : length(m{j})
        X_tst = [X_tst; reshape(data_set1(m{j}(p),:),1,length(freq))];
        Y_tst = [Y_tst; 1];
    end
    for p = 1 : length(n{j})
        X_tst = [X_tst; reshape(data_set2(n{j}(p),:),1,length(freq))];
        Y_tst = [Y_tst; 0];
    end

    X_trn = []; Y_trn = [];
    for p = 1 : length(mb{j})
        X_trn = [X_trn; reshape(data_set1(mb{j}(p),:),1,length(freq))];
        Y_trn = [Y_trn; 1];
    end
    for p = 1 : length(nb{j})
        X_trn = [X_trn; reshape(data_set2(nb{j}(p),:),1,length(freq))];
        Y_trn = [Y_trn; 0];
    end          
    
    extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)
    model = classRF_train(X_trn,Y_trn, trees, nfeat, extra_options);
    [Y_hat, votes] = classRF_predict(X_tst,model);
    acc_fold(j) = length(find(Y_hat == Y_tst))/length(Y_tst);
    gini(j,:) = model.importance(:,end);
    
end
Acc = mean(acc_fold); % mean accuracy from 3 crossvalidations

end

