function [her, data_trials, trials] = compute_her(data_eeg, ibi, indx, time_onset, time_offset, maxamp)
% This function computes timelocked EEG segments to R peak timings, namely
% Heartbeat-evoked response. Are discarded EEG epochs overlaped and the
% ones with amplitude higher than maxamp. This function requires Fieldtrip
% toolbox.
% Inputs:
% data_eeg: Fieldtrip data struct of a single subject with preprocessed EEG
% ibi: array with interbeat intervals duration (seconds)
% indx: array with sample information of R peaks, w.r.t. the time in data_eeg
% time onset: EEG epoch onset respect the R peak timing (seconds)
% time offset: EEG epoch offset respect the R peak timing (seconds)
% maxamp: uV max amplitude for good trials in EEG epochs.
%
% Outputs:
% her: Fieldtrip struct with timelocked data
% data_trials: Eieldtrip struct with EEG epochs as trials
% trials: array with trial numbers used to perform the timelock average
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

%% Get EEG epochs
% define interval
sample_onset = floor(time_onset/1000 * data_eeg.fsample);
sample_offset = floor(time_offset/1000 * data_eeg.fsample);

% redefine trials: 1 HER = 1 trial
begsample = indx(1:end-1)' + sample_onset;
endsample = indx(1:end-1)' + sample_offset;
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
trials_rejected2 = find(ibi < time_offset/1000)';
trials_rejected = unique([trials_rejected1 trials_rejected2]);
trials = setdiff(1:length(ibi),trials_rejected);

%% Timelock epochs
cfg=[];
cfg.trials = trials;
cfg.removemean = 'no';
her = ft_timelockanalysis(cfg, data_trials);

end

