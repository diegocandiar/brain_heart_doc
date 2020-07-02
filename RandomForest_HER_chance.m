function [acc_rand] = RandomForest_HER_chance(set_avg, set_var, numrand, train1, train2, test1, test2, time_onset, time_offset)
% This function performs a Random Forest classification using arbitrary
% patients label. This function performs several classifications to obtain a
% chance accuracy distribution for unbalanced datasets. This function uses
% Random Forest console as implemented in: https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% set_avg: fieldtrip struct with all patients data. 1 patient average timelock = 1 trial
% set_var: fieldtrip struct with all patients data. 1 patient variance timelock = 1 trial
% numrand: number of classifications to perform
% train1: number of patients considered as class1 in training set
% train2: number of patients considered as class2 in training set
% test1: number of patients considered as class1 in test set
% test2: number of patients considered as class2 in test set
% time onset: EEG epoch onset respect the R peak timing (seconds)
% time offset: EEG epoch offset respect the R peak timing (seconds)
% Outputs:
% acc_rand: chance accuracy distribution
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX


%% Averaging time windows
L = length(set_avg.trial);
ch = length(set_avg.trial(:,1));
freq = set_avg.fsample;
time = set_avg.time; % time array

ts = 30; %time window in miliseconds
ov = 0.8; % percentage of overlap between timewindows
ss = ts/1000*freq; %time window in number of samples
sc = ss*ov;
tt = ceil((length(time)-ss)/sc); % final number of samples after the reduction
time = zeros(1,tt); % new time array

for i = 1:L % iteration for all patients
    for k = 1:ch % iteration for all channels
        for j = 1:tt % iteration for all of the final samples
            tw = ((j-1)*sc+1) : (((j-1)*sc)+ss);
            time(j) = median(tw)/freq;
            M1{1,i}(k,j) = mean(set_avg.trial{1,i}(k,tw));
            M1b{1,i}(k,j) = mean(set_var.trial{1,i}(k,tw));
        end
    end
set_avg.time{1,i} = time; 
set_var.time{1,i} = time; 
end

set_avg.trial = M1;
set_var.trial = M1b;

%% perform randomizations
acc_rand = zeros(1,numrand);

for n_rand = 1:numrand
  
    %% random sort
    patients = randperm(L);
    
    % random patients distribution
    set1_train = patients(1 : train1);
    set2_train = patients(train1+1 : train1+train2);   
    set1_test = patients(train1+train2+1 : train1+train2+test1);
    set2_test = patients(train1+train2+test1+1 : train1+train2+test1+test2);
    
    TEST = [set1_test set2_test];

    % set1
    cfg = [];
    cfg.trials = set1_train;
    set1_avg = ft_selectdata(cfg,set_avg);
    set1_var = ft_selectdata(cfg,set_var);
    
    % set2
    cfg = [];
    cfg.trials = set2_train;
    set2_avg = ft_selectdata(cfg,set_avg);
    set2_var = ft_selectdata(cfg,set_var);   
    
    %TEST
    cfg = [];
    cfg.trials = TEST;
    set3_avg = ft_selectdata(cfg,set_avg);
    set3_var = ft_selectdata(cfg,set_var);

    %% arrange datasets

    i1 = find(abs(time-time_onset) == min(abs(time-time_onset)));
    i2 = find(abs(time-time_offset) == min(abs(time-time_offset)));
    sp = i1:i2;
    
    data_set1_avg = zeros(L1, ch* length(sp));
    data_set1_var = zeros(L1, ch* length(sp));
    for i = 1:L1
        data_set1_avg(i,:) = reshape(set1_avg.trial{i}(ch,sp),1,ch*length(sp));
        data_set1_var(i,:) = reshape(set1_var.trial{i}(ch,sp),1,ch*length(sp));
    end

    data_set2_avg = zeros(L1, ch* length(sp));
    data_set2_var = zeros(L1, ch* length(sp));
    for i = 1:L2
        data_set2_avg(i,:) = reshape(set2_avg.trial{i}(ch,sp),1,ch*length(sp));
        data_set2_var(i,:) = reshape(set2_var.trial{i}(ch,sp),1,ch*length(sp));
    end

    data_set3_avg = zeros(L1, ch* length(sp));
    data_set3_var = zeros(L1, ch* length(sp));
    for i = 1:L1
        data_set3_avg(i,:) = reshape(set3_avg.trial{i}(ch,sp),1,ch*length(sp));
        data_set3_var(i,:) = reshape(set3_var.trial{i}(ch,sp),1,ch*length(sp));
    end

    feat = 1:2*ch*length(sp);
    data_set1 = [data_set1_avg data_set1_var];
    data_set2 = [data_set2_avg data_set2_var];
    data_set3 = [data_set3_avg data_set3_var];

    %% Random Forest
    nfeat = max(2,floor(length(feat)^0.5));
    trees = 1000;
    X_trn = [data_set1(:,feat); data_set2(:,feat)];
    Y_trn = [ones(length(set1_train),1); zeros(length(set2_train),1)];
    X_tst = [data_set3(:,feat)];
    Y_tst = [ones(test1,1); zeros(test2,1)];

    model = classRF_train(X_trn,Y_trn, trees, nfeat);

    [Y_hat, votes] = classRF_predict(X_tst,model);
    acc_rand(n_rand) = length(find(Y_hat' == Y_tst))/length(Y_tst);
    
end


