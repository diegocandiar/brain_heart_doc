function [Acc, time_acc] = RandomForest_HER_crossvalidation_sliding(set1_avg, set1_var, set2_avg, set2_var, nfolds, window)
% This function performs a Random Forest crossvalidation classification.
% Classifications are performed with a sliding time window
% This function divides the set of patients in folds to perform
% classifications using different set of patients each time. This function
% uses Random Forest console as implemented in:
% https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% set1_avg: fieldtrip struct with all patients data of class1. 1 patient average timelock = 1 trial
% set1_var: fieldtrip struct with all patients data of class1. 1 patient variance timelock = 1 trial
% set2_avg: fieldtrip struct with all patients data of class2. 1 patient average timelock = 1 trial
% set2_var: fieldtrip struct with all patients data of class2. 1 patient variance timelock = 1 trial
% nfolds: number of folds (integer)
% window: sliding time window length in seconds
% Outputs:
% Acc: crossvalidation accuracy array
% time_acc: array with time centers of each crossvalidation performed
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX


%% Compile Random Forest
compile_windows

%% read data
L1 = length(set1_avg.trial);
L2 = length(set2_avg.trial);
ch = length(set1_avg.trial(:,1));
freq = set1_avg.fsample;
time = set1_avg.time{1};

%% Averaging time windows

ts = 30; %time window in miliseconds
ov = 0.8; % percentage of overlap between timewindows
ss = ts/1000*freq; %time window in number of samples
sc = ss*ov;
tt = ceil((length(time)-ss)/sc); % final number of samples after the reduction
time = zeros(1,tt); % new time array

% set 1
for i = 1 : L1 % iteration for all set1 patients
    for k = 1:ch % iteration for all channels
        for j = 1:tt % iteration for all of the final samples
            tw = ((j-1)*sc+1) : (((j-1)*sc)+ss);
            time(j) = median(tw)/freq;
            M1{1,i}(k,j) = mean(set1_avg.trial{1,i}(k,tw));
            M1b{1,i}(k,j) = mean(set1_var.trial{1,i}(k,tw));
        end
    end
set1_avg.time{1,i} = time; 
set1_var.time{1,i} = time; 
end

% set 2
for i = 1 : L2
    for k = 1:ch
        for j = 1:tt
            tw = ((j-1)*sc+1) : (((j-1)*sc)+ss);
            time(j) = median(tw)/freq;
            M2{1,i}(k,j) = mean(set2_avg.trial{1,i}(k,tw));
            M2b{1,i}(k,j) = mean(set2_var.trial{1,i}(k,tw));
        end
    end
set2_avg.time{1,i} = time; 
set2_var.time{1,i} = time; 
end

% Combine avg and var in one struct
set1 = struct; set1.label = set1_avg.label;
set2 = struct; set2.label = set1_avg.label;
for i = 1:length(time)
    for j = 1:L1
        k = (i-1)*2+1;
        set1.trial{1,j}(:,k) = M1{1,j}(:,i);
        set1.trial{1,j}(:,k+1) = M1b{1,j}(:,i);
    end
    
    for j = 1:L2
        k = (i-1)*2+1;
        set2.trial{1,j}(:,k) = M2{1,j}(:,i);
        set2.trial{1,j}(:,k+1) = M2b{1,j}(:,i);
    end
end

for j = 1:L1
    set1.time{1,j} = 1:length(time)*2;
end

for j = 1:L2
    set2.time{1,j} = 1:length(time)*2;
end

%% Random Forest config
time_aux = abs(window-(time-time(1)));
nsamp = find(time_aux == min(time_aux)); % number of timestamps to enter
nfeat = floor((nsamp*ch*2)^0.5); % number of features in each node
trees = 1000; % total trees in random forest

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

%% Perform random forest: iterate for sliding time window classification
nclass = tt-nsamp+1;
Acc = zeros(1, nclass);
time_acc = zeros(1, nclass);

for i = 1:nclass
    k = (i-1)*2+1;
    acc = zeros(1,nfolds); % store fold accuracies
    for j = 1:nfolds
        % prepare test set
        X_tst = []; Y_tst = [];
        for p = 1 : length(m{j})
            data_tmp = set1.trial{1,m{j}(p)}(:, k : (k+nsamp*2 - 1));
            X_tst = [X_tst; reshape(data_tmp',1,nsamp*ch*2)];
            Y_tst = [Y_tst; 1];
        end
        for p = 1 : length(n{j})
            data_tmp = set2.trial{1,n{j}(p)}(:, k : (k+nsamp*2 - 1));
            X_tst = [X_tst; reshape(data_tmp',1,nsamp*ch*2)];
            Y_tst = [Y_tst; 0];
        end
        %prepare train set
        X_trn = []; Y_trn = [];
        for p = 1 : length(mb{j})
            data_tmp = set1.trial{1,mb{j}(p)}(:, k : (k+nsamp*2 - 1));
            X_trn = [X_trn; reshape(data_tmp',1,nsamp*ch*2)];
            Y_trn = [Y_trn; 1];
        end
        for p = 1 : length(nb{j})
            data_tmp = set2.trial{1,nb{j}(p)}(:, k : (k+nsamp*2 - 1));
            X_trn = [X_trn; reshape(data_tmp',1,nsamp*ch*2)];
            Y_trn = [Y_trn; 0];
        end            
        % training    
        model = classRF_train(X_trn,Y_trn, trees, nfeat);
        % test
        [Y_hat, votes] = classRF_predict(X_tst,model);
        acc(j) = length(find(Y_hat == Y_tst))/length(Y_tst);
    end
    Acc(i) = mean(acc);
    time_acc(i) = mean(time(k : (k+nsamp*2 - 1)));
end

%% visualize

figure
area(time_acc*1000,Acc,'FaceColor',[0.3 0.75 0.93])
title('Accuracy obtained using single timestamp')
ylabel('Accuracy')
xlabel('Time (ms)')
ylim([0.7 0.95])
xlim([120 time_acc(end)*1000])
set(gca,'fontsize',12)
set(findall(gcf,'type','text'),'FontSize',13)
set(gcf,'units','points','position',[10,10,800,200])

end

