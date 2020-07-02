function [eeg_timelock, data_trials, trials, indx2, ibi2, overlap] = compute_eeg_segments(data_eeg, indx, time_onset, time_offset, maxamp)
% This function computes timelocked EEG segments to random timings. This
% function emulates what is done in the function compute_her.m and random
% timings are considered as heartbeats. Are discarded EEG epochs overlaped
% and the ones with amplitude higher than maxamp.
% This function requires Fieldtrip toolbox.
% Inputs:
% data_eeg: Fieldtrip data struct of a single subject with preprocessed EEG
% indx: array with sample information of R peaks, respect the array time
% time: time array
% time onset: EEG epoch onset respect the surrogate R peak timing (seconds)
% time offset: EEG epoch offset respect the surrogate R peak timing (seconds)
% maxamp: uV max amplitude for good trials in EEG epochs.
%
% Outputs:
% eeg_timelock: Fieldtrip struct with timelocked data
% data_trials: Eieldtrip struct with EEG epochs as trials
% trials: array with trial numbers used to perform the timelock average
% indx2: array with sample information of surrogate R peaks, respect the array time
% ibi2: surrogate ibi array (used to reject trials)
% overlap: Overlap fraction of surrogate EEG epochs respect real HER epochs.
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX


%% Randomize heartbeats
indx2 = zeros(1,length(indx));

% random each onset
for i = 1:length(indx2)
    indx2(i) = randi([1 length(time)]);
end

indx2 = sort(indx2);
ibi2 = diff(time(indx2));


%% Get EEG epochs
% define interval
sample_onset = floor(time_onset* data_eeg.fsample);
sample_offset = floor(time_offset * data_eeg.fsample);

% redefine trials: 1 HER = 1 trial
begsample = indx2(1:end-1)' + sample_onset;
endsample = indx2(1:end-1)' + sample_offset;
trigger = zeros(length(begsample),1);
trl = [begsample endsample trigger];
cfg = [];
cfg.trl = trl;
data_trials = ft_redefinetrial(cfg, data_eeg);

% resample to 250 Hz (for Fs 500 Hz only)
if data_eeg.fsample ~= 250
    cfg = [];
    cfg.resamplefs = 250;
    data_trials = ft_resampledata(cfg, data_trials);
end

%% Epoch selection
% Select trials 1: based on large artefacts 
trials_rejected1 = 0;
for i = 1:length(data_trials.trial)
    cfg = [];
    cfg.trials = i;
    data_temp = ft_selectdata(cfg, data_trials);
    if max(max(data_temp.trial{1}(:,1:end))) > maxamp || min(min(data_temp.trial{1}(:,1:end))) < -maxamp
        trials_rejected1 = [trials_rejected1 i];
    end  
end

% Select trials 2: based on short IBI
trials_rejected2 = find(ibi2 < time_offset/1000)';
trials_rejected = unique([trials_rejected1 trials_rejected2]);
trials = setdiff(1:length(ibi2),trials_rejected);

%% Timelock epochs
cfg=[];
cfg.trials = trials;
cfg.removemean = 'no';
eeg_timelock = ft_timelockanalysis(cfg, data_trials);

%% compute overlap of new eeg epochs with real her

%quantify overlap
fs = data_eeg.fsample;
sample_window_onset = time_onset* fs;
sample_window_offset = time_offset* fs;

sample_window = sample_window_offset - sample_window_onset;

% get real HER timings
her_real = zeros(1,sample_window*length(indx));
for j = 1:length(indx)
    her_real( ((j-1)*sample_window+1) : (j*sample_window) ) = (indx(j)+sample_window_onset):(indx(j)+sample_window_offset-1);
end
if length(her_real) > length(time)
    her_real( (length(time)+1) : length(her_real) ) = [];
end

% get EEG epochs timings
her_rand = zeros(1,sample_window*length(indx));
for j = 1:length(indx2)
    her_rand( ((j-1)*sample_window+1) : (j*sample_window) ) = (indx2(j)+sample_window_onset):(indx2(j)+sample_window_offset-1);
end
if length(her_rand) > length(time)
    her_rand( (length(time)+1) : length(her_rand) ) = [];
end

% quantify overlap
overlap = length(intersect(her_real,her_rand))/length(her_rand);

end

