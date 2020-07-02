function [Acc, Y_hat, Score, Gini, features_names] = RandomForest_HER_validation(set1_avg, set1_var, set2_avg, set2_var, set3_avg, set3_var, Y_tst, time_onset, time_offset)
% This function performs a Random Forest validation classification. This
% function uses Random Forest console as implemented in:
% https://code.google.com/archive/p/randomforest-matlab/
%
% Inputs:
% set1_avg: fieldtrip struct with training patients data of class1. 1 patient average timelock = 1 trial
% set1_var: fieldtrip struct with training patients data of class1. 1 patient variance timelock = 1 trial
% set2_avg: fieldtrip struct with training patients data of class2. 1 patient average timelock = 1 trial
% set2_var: fieldtrip struct with training patients data of class2. 1 patient variance timelock = 1 trial
% set3_avg: fieldtrip struct with test patients data. 1 patient average timelock = 1 trial
% set3_var: fieldtrip struct with test patients data. 1 patient variance timelock = 1 trial
% Y_test: logical array with test set class. 1 = class1, 0= class2
% time onset: EEG epoch onset respect the R peak timing (seconds)
% time offset: EEG epoch offset respect the R peak timing (seconds)
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

%% Compile Random Forest
compile_windows

%% read data
L1 = length(set1_avg.trial);
L2 = length(set2_avg.trial);
L3 = length(set3_avg.trial);
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

% set 3
for i = 1 : L3
    for k = 1:ch
        for j = 1:tt
            tw = ((j-1)*sc+1) : (((j-1)*sc)+ss);
            time(j) = median(tw)/freq;
            M3{1,i}(k,j) = mean(set3_avg.trial{1,i}(k,tw));
            M3b{1,i}(k,j) = mean(set3_var.trial{1,i}(k,tw));
        end
    end
set3_avg.time{1,i} = time; 
set3_var.time{1,i} = time; 
end

% Combine avg and var in one struct
set1 = struct; set1.label = set1_avg.label;
set2 = struct; set2.label = set1_avg.label;
set3 = struct; set3.label = set1_avg.label;
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
    for j = 1:L3
        k = (i-1)*2+1;
        set3.trial{1,j}(:,k) = M3{1,j}(:,i);
        set3.trial{1,j}(:,k+1) = M3b{1,j}(:,i);
    end
end

for j = 1:L1
    set1.time{1,j} = 1:length(time)*2;
end

for j = 1:L2
    set2.time{1,j} = 1:length(time)*2;
end

for j = 1:L3
    set3.time{1,j} = 1:length(time)*2;
end

%% define timestamps to use and feature matrix
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

data_set1 = [data_set1_avg data_set1_var];
data_set2 = [data_set2_avg data_set2_var];
data_set3 = [data_set3_avg data_set3_var];

%% features names
features_names1 = {};
labels = set1_avg.label;
k = 1;
for i = 1:length(sp)
    for j = 1:ch
        features_names1{k} = strcat(labels{j},sprintf(' %d ms avg',time(sp(i))*1000));
        k = k+1;
    end
end

features_names2 = {};
k = 1;
for i = 1:length(sp)
    for j = 1:ch
        features_names2{k} = strcat(labels{j},sprintf(' %d ms var',time(sp(i))*1000));
        k = k+1;
    end
end
features_names = [features_names1 features_names2];

%% random forest parameters
feat = 1:2*ch*length(sp);
nfeat = max(2,floor(length(feat)^0.5));
trees = 1000;

%% Random Forest

X_trn = [data_set1(:,feat); data_set2(:,feat)];
Y_trn = [ones(L1,1); zeros(L2,1)];
X_tst = [data_set3_avg(:,feat)];

extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)
extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculate)
model = classRF_train(X_trn,Y_trn, trees, nfeat, extra_options);

[Y_hat, votes] = classRF_predict(X_tst,model);

Acc = length(find(Y_hat == Y_tst))/length(Y_hat);

%% Compute consciousness scores

Score = zeros(1, L3);
for i = 1:length(Y_hat)
    Score(i) = votes(i,2)/trees;
end

%% Get feature importance
n_feat_imp = min(10, length(features_names));

Gini = model.importance(:,end);
[gini_sort, indx] = sort(Gini,'descend');
feat = indx(1:n_feat_imp);

figure
bar(model.importance(:,end), 'FaceColor',[0 0.75 .25]); 
title('Feature importance - Gini index'); ylabel('magnitude');
xlim([0 length([feat])+1])
set(gcf,'units','points','position',[10,10,1200,400])
xticklabel_rotate(1:length(feat),45,features_names(feat)')

end

