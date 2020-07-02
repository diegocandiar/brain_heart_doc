function [data_eeg, ICA_ECG] = processing_eeg(raw_data_path, elec_positions, neighbours)
% This code performs EEG preprocessing including frequency filter, bad
% channels detection and ICA computation. It reads the raw data from the
% specified path. Code adapted for EEG EGI 256 channels + ref channel. 
% This function requires Fieldtrip toolbox
% Inputs:
% raw_data_path: raw data location 
% elec_positions: fieldtrip struct with electrodes information
% neighbours: fieldtrip struct with electrodes neighbours information
%
% Outputs:
% data_eeg: preprocessed EEG data
% ICA_ECG: array with selected ICA component corresponding to ECG artefacts
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

%% read raw data
% header raw data
hdr = ft_read_header(raw_data_path); %data header

%% Frequency filter
cfg=[];
cfg.dataset = raw_data_path;
cfg.bpfilter = 'yes';
cfg.bpfreq = [1 25]; % bandpass filter limits, in Hz
data = ft_preprocessing(cfg);
n_channels = length(data.labels);

%% Define trials. Make multiple segments from the continuos recording
timewindow = 5*60; %time window in seconds
samplewindow = timewindow*hdr.Fs;
time_step = 10; % step in seconds
step = time_step*hdr.Fs;
o1 = []; %begin trial
o2 = []; %end trial
j = 0;
for i = 1:step:(hdr.nSamples-samplewindow) 
    j = j + 1;
    o1(j) = i; 
    o2(j) = o1(j)+samplewindow-1;
end

%% Quantify artifacts in all trials 
artifact_duration  = zeros(1,length(o1)); % quantify artifact duration (sum)
number_artifact  = zeros(1,length(o1)); % quantify artifacts
chan_bad_zscore = {};
chan_good_zscore = {};

for i = 1:length(o1)
    %% redefine trial
    cfg=[];
    cfg.begsample = o1(i);
    cfg.endsample = o2(i);
    data_temp = ft_redefinetrial(cfg, data);

    %% bad channels, criteria 1: z-score distribution
    isbad1 = []; % bad channels array 1st criteria
    good_ch1 = 1:n_channels;
    while length(isbad1) <= 50 %iteration, stop criteria: 50 bad channels or no more bad channels
        % iteration for computing area under the curve of each channel
        z_density = [];
        for j = 1:length(good_ch1)
            chan_data    = data_temp.trial{1,1}(good_ch1(j),:);
            z_density(j) = sum(abs(chan_data));
        end
        % iteration for rejecting channels higher than 3 SD
        a = length(isbad1);
        for j = 1:length(z_density)
            if abs(z_density(j)) >=  mean(z_density) + 3*std(z_density)
                isbad1 = [isbad1,good_ch1(j)];
            end
        end
        % update bad/good channels arrays
        [~,~,ib] = intersect(isbad1,good_ch1) ;
        good_ch1(ib) = [];
        if a == length(isbad1)
            break;
        end
    end
    % store bad/good channels for the current trial
    chan_bad_zscore{i} = isbad1;
    chan_good_zscore{i} = good_ch1;
    %% quantify noisy segments
    % every trial will have a total duration of very large artefacts
    cfg=[];
    cfg.continuous = 'yes';
    cfg.artfctdef.zvalue.channel = good_ch1; % use good channels after z-score criteria
    cfg.artfctdef.zvalue.cutoff = 20; % z-score threshold =4 default fieldtrip
    [~, artifact] = ft_artifact_zvalue(cfg, data_temp);
    number_artifact(i) = length(artifact);
    length_artifact = 0;
    for j = 1:length(artifact(:,1))
        length_artifact = length_artifact + (artifact(j,2) - artifact(j,1));
    end
    artifact_duration(i) = length_artifact;    
end
% ranking of less noisy trials, based on total duration of large artefacts
[~, trials_rank] = sort(artifact_duration); 
% select less noisy trial
cfg=[];
cfg.begsample = o1(trials_rank(1));
cfg.endsample = o2(trials_rank(1));
% update based on the selected trial
data_temp = ft_redefinetrial(cfg, data);
isbad1 = chan_bad_zscore{trials_rank(1)};
good_ch1 = chan_good_zscore{trials_rank(1)};

%% Detect bad channels, criteria 2: weighted-correlation of neighbours
isbad_label = data_temp.label(isbad1);
data_temp.label(1:n_channels)=elec_positions.label(1:n_channels);

%iteration to delete from neighbours definition the bad channels from
%criteria 1
for i = 1:n_channels
    for k = 1:length(isbad1)
        for j = 1:length(neighbours(1,i).neighblabel)
            if strcmp(neighbours(1,i).neighblabel{j,1},isbad_label{k,1}) == 1
                neighbours(1,i).neighblabel{j,1} = [];
            end
        end
    end
    neighbours(1,i).neighblabel(all(cellfun(@isempty,neighbours(1,i).neighblabel),2), : ) = [];
end

%compute correlation coefficients
isbad2 = [];
cfg = [];
cfg.method          = 'weighted';
cfg.neighbours      = neighbours;
cfg.trials          = 'all';
cfg.elec            = elec;
corrcoefs = zeros(1,length(good_ch1));
for i = 1:length(good_ch1)
    cfg.badchannel = data_temp.label(good_ch1(i));
    interp = ft_channelrepair(cfg, data_temp);
    R = corrcoef(interp.trial{1,1}(good_ch1(i),:),data_temp.trial{1,1}(good_ch1(i),:));
    corrcoefs(i) = R(1,2);
    if corrcoefs(i) < 0.8 || isnan(corrcoefs(i))
        isbad2 =[isbad2,good_ch1(i)];
    end
end
if (length(isbad2)+length(isbad1)) > 140 % superior threshold of bad channels
    fprintf('Again too many bad channels! Just the worst 140 will be discarded \n')
    isbad2 = [];
    isbad2 = find(isnan(corrcoefs));
    [Y,I] = sort(corrcoefs,'ascend');
    isbad2 = [isbad2 good_ch1(I(1:(140-length(isbad2)-length(isbad1))))];
end    
isbad = [isbad1 isbad2];
good_ch = 1:n_channels;
good_ch(isbad) = [];

%% ICA
cfg = [];
cfg.channel = good_ch;
cfg.method = 'runica'; % Infomax ICA
cfg.runica.extended = 1; 
cfg.runica.stop = 1e-7;
cfg.runica.maxsteps = 500;
ICA = ft_componentanalysis(cfg, data_temp);

%% Find ICA-ECG visually
cfg = [];
cfg.viewmode = 'component';
cfg.layout = layout;
ft_databrowser(cfg, ICA)  

comp_ecg = input('Enter ICA-ECG components numbers (clearest one first if more than one): ');
ICA_ECG = ICA.trial{1}(comp_ecg(1),:);

%% REMOVE HEART ARTEFACTS (cardiac artefacts components from ICA)
cfg = [];
cfg.component = comp_ecg;
data_eeg = ft_rejectcomponent(cfg,ICA,data_temp);

%% start from t = 0
data_eeg.time = data_eeg.time - data_eeg.time{1,1}(1);
data_eeg.sampleinfo = [1,data_eeg.fsample*300];

%% add Cz channel as bad channel for later interpolation (ref channel)
data_eeg.label{257,1} = 'Cz';
data_temp.label{257,1} = 'Cz';
data_eeg.trial{1,1}(257,:) = zeros(data_eeg.fsample*300,1);
data_temp.trial{1,1}(257,:) = zeros(data_temp.fsample*300,1);
isbad = [isbad 257];

%% INTERPOLATE BAD CHANNELS
% Define channels to use (175 channels, excluding face and neck channels)
channels = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 95, 96, 97, 98, 99, 100, 101, 106, 107, 108, 109, 110, 115, 115, 116, 117, 118, 119, 124, 125, 126, 127, 128, 129, 130, 131, 132, 137, 138, 139, 140, 141, 142, 143, 144, 149, 150, 151, 152, 153, 154, 155, 159, 160, 161, 162, 163, 164, 169, 170, 171, 172, 173, 178, 179, 180, 181, 182, 183, 184, 185, 186, 191, 192, 193, 194, 195, 196, 197, 198, 202, 203, 204, 205, 206, 207, 210, 211, 212, 213, 214, 215, 220, 221, 222, 223, 224, 257];

all_channels = 1:257;
bad_channels = setdiff(all_channels,channels);
bad_channels = unique([bad_channels isbad]);
bad_labels = data_temp.label(bad_channels);

%Discard bad neighbors (removes bad channels from neighbours' definition from
%all channels.
for i = 1:length(all_channels)
    for k = 1:length(bad_channels)
        for j = 1:length(neighbours(1,i).neighblabel)
            if strcmp(neighbours(1,i).neighblabel{j,1},bad_labels{k,1}) == 1
                neighbours(1,i).neighblabel{j,1} = [];
            end
        end
    end
    neighbours(1,i).neighblabel(all(cellfun(@isempty,neighbours(1,i).neighblabel),2), : ) = [];
end

% Interpolate bad channel using only good channels
cfg = [];
cfg.method = 'spline';
cfg.badchannel = data_temp.label(isbad);
cfg.neighbours = neighbours;
cfg.elec = elec_positions;
data_eeg = ft_channelrepair(cfg,data_eeg);

%% re-reference using common avergae (of the previously defined 175 channels)
cfg = [];
cfg.reref = 'yes';
cfg.refchannel = channels;
data_eeg = ft_preprocessing(cfg, data_eeg);

end

