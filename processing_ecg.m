function [ibi, indx, peak_indx, ADD, REM] = processing_ecg(ECG, time, fsample)
% This code performs heartbeats detection from ECG or ICA-ECG. The
% procedure includes heartbeat template, peaks
% detection and manual edition of misdetections.
% This function requires the function editpeaks.m
% Inputs:
% ECG: ECG array
% time: time array respect ECG
% fsample: sample frequency
%
% Outputs
% ibi: array with interbeat intervals duration in seconds
% indx: array with sample information of R peaks, respect the array time
% peak_indx: logical array with R peaks location, respect the array time
% ADD: array with sample information of added R peaks manually
% REM: array with sample information of removed R peaks manually
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

%% Modify polarity if needed
if sign(skewness(ECG)) == -1
    ECG = -ECG;
end

%% Find time window to create R peak template

% user defined time window
s1 = input('Press <y> if you found a suitable time window: ', 's');
if strcmp(s1,'y')
    title('Select time window ONSET with mouse')
    [onset,~] = ginput(1);
    title('Select time window OFFSET with mouse')
    [offset,~] = ginput(1);
end

% extract the ecg data that correspond to the time window selected by the
% user:
ecg_tw = ECG(onset:offset);
time_tw = ECG(onset:offset);

% detect peaks time window
prominence = mean(ecg_tw)+3*std(ecg_tw);
mindist = 0.45 * fsample;
[~,pks] = findpeaks(ecg_tw, 'MinPeakProminence', prominence, 'MinPeakDistance', mindist);

%% edit peaks template

ADD =[]; REM=[];
figure
plot(ecg_tw); hold on; plot(pks, ecg_tw(pks), 'ok')
pks_bin = zeros(1, length(time_tw)); pks_bin(pks) = 1;
s1 = input('Would you like to modify heartbeats manually (y/n)? ', 's');
while strcmp(s1,'y')
    s3 = input('Would you like to add/remove peaks (a/r/n) (a = add / r = remove / n = no)? ', 's');
    while any(strcmp(s3,{'a' 'r'}))
        title('Click with mouse to add or remove peaks')
        [ol,au] = ginput(1);
        ol = round(ol);

        % remove the selected peak
        if strcmp(s3,'r')
            REM = [REM ol];
            display('Deleting peak...')
            hold on
            % mark the deleted peak
            % plot(ECG_trial);
            hold on
            plot(ol,au,'or','markersize',8)
            if ol - 5 <= 0 % this takes care of the start of the window
                pks_bin(1:ol+5) = 0;
            elseif ol + 5 > length(pks_bin) % this takes care of the end of the window
                pks_bin(ol-5:end) = 0;
            else
                pks_bin(ol-5:ol+5) = 0;
            end

        % add the desired peak
        elseif strcmp(s3,'a')
            ADD = [ADD ol];
            display('Adding peak...')
            % plot the added peak
            % plot(ECG_trial);
            hold on
            plot(ol,au,'og','markersize',8)
            if ol - 5 <= 0 % this takes care of the start of the window
                [~,mx] = max(ecg_tw(1:ol+5));
                pks_bin(mx) = 1;
            elseif ol + 5 > length(pks_bin) % this takes care of the end of the window
                [~,mx] = max(ecg_tw(ol-5:end));
                pks_bin(mx + ol -5 - 1) = 1;
            else
                [~,mx] = max(ecg_tw(ol-5:ol+5));
                pks_bin(mx + ol -5 - 1) = 1;
            end
        end
        s3 = input('Would you like to add/remove further peaks (a/r/n) (a = add / r = remove / n = no)? ', 's');
    end
    pks = find(pks_bin == 1);
    figure
    plot(ecg_tw); hold on; plot(pks, ecg_tw(pks), 'ok')
    s1 = input('Would you like to modify heartbeats manually (y/n)? ', 's');
end

%% Compute mean and std IBI from template window
IBI_tw = time_tw(pks);
ratemean = mean(IBI_tw);
ratestd = std(IBI_tw);

%% build template
template_r=[];
template_tw = 0.4 * fs; % length template
if length(pks)==1 % if only one peak has been detected, the template is just the peak
    template_r=ecg_tw;
else% in alll other cases, it is the mean of the peaks detected
    for ii=1:length(pks)
        if pks(ii)-template_tw/2 > 1 && pks(ii)+template_tw/2 < length(ecg_tw)
            template_r(ii,:) = ecg_tw(pks(ii)-template_tw/2:pks(ii)+template_tw/2);
        end
    end
    template_r = mean(template_r);
end

%% perform correlation using template

ECG_pad = [zeros(1,1000) ECG zeros(1,1000)];
cr = zeros(size(ECG_pad));
for i=1:length(ECG_pad)-length(template_r)
    cr(i+round(length(template_r)/2)-1) = sum(ECG_pad(i:i+length(template_r)-1).*template_r);
end
ECG_corr = cr(1001:end-1000)/max(cr); % normalize correlation to 1

%% Detect R-peaks on ECG correlation

M = 2*max(ECG_corr);
ECGm = ECG_corr;
midwind = round(freq*(ratemean+0.2)/2);
wind = 2*midwind;

delta = 5;
% find local maxima
for i = 1:length(ECGm)
    if i - midwind <= 1 % this takes care of the start of the window
        o1 = i;
        o2 = o1 + wind;
        ECG_window = ECGm(o1:o2);
        [~, Mi] = max(ECG_window);
        j = Mi(1) + o1 - 1;
        if j - delta <= 0 % this takes care of the start of the window
            [~,mx] = max(ECG(1:j+delta));  
        else
            [~,mx] = max(ECG(j-delta:j+delta));
            mx = mx(1) + j - delta - 1;
        end        
        [~,mx] = max(ECG(1:j+delta));
    elseif i + midwind >= length(ECG) % this takes care of the end of the window
        o1 = i - midwind ;
        o2 = length(ECG);
        ECG_window = ECGm(o1:o2);
        [~, Mi] = max(ECG_window);
        j = Mi(1) + o1 - 1;
        if j + delta > length(ECG) % this takes care of the end of the window
            [~,mx] = max(ECG(j-delta:end));  
        else
            [~,mx] = max(ECG(j-delta:j+delta));
            mx = mx(1) + j - delta - 1;
        end        
    else
        o1 = i-midwind;
        o2 = i+midwind;
        ECG_window = ECGm(o1:o2);
        [~, Mi] = max(ECG_window);
        j = Mi(1) + o1 -1;
        [~,mx] = max(ECG(j-delta:j+delta));
        mx = mx(1) + j - delta - 1;
    end
    ECGm(mx) = M;
end

% store indexes of possible R peaks
indx = [];
for i = 1:length(ECGm)-1
    if ECGm(i) == M
        indx = [indx i];
    end
end

% discard consecutive peaks
m = ECGm(indx);
d = diff(indx);
jump = (d> (ratemean-0.2)*freq);
p = [];

sect=1;
while sect<=length(d)
    if jump(sect)
        p = [p indx(sect)];
    else
        s = [];
        while ~jump(sect) & sect<length(d)
            s = [s sect];
            sect = sect + 1;
        end
        [lm, li] = max(m(s));
        p = [p indx(s(li))];
    end
    sect = sect+1;
end
indx = p;
ibi = diff(time(indx));

%% Visualize detection
close all
figure
hist(ibi,20,'Color',[0.75 0 0.75])
title('Heart-Rate Histogram')
xlabel('Time [s]')
ylabel('Frequency')

figure
ECGplot1 = ECG_corr*15;
ECGplot1(ECGplot1>5) = 5;
plot(ECGplot1)
hold on
plot(indx,ECGplot1(indx),'ok')
hold on
ECGplot2 = zscore(ECG)+10;
ECGplot2(ECGplot2>15) = 15;
plot(ECGplot2)
hold on
plot(indx,ECGplot2(indx),'ok')
title('EEG signal with ECG artifacts')
xlabel('[samples]')
ylabel('[uV]')
ylim([0 15])
set(gcf,'units','points','position',[10,10,1200,300])

%% perform peaks editing

% save peaks added and removed
ADD = []; % samples where heartbeats were added
REM = []; % samples where heartbeats were removed
peak_indx = zeros(1,length(ECG));
peak_indx(indx) = 1;

% iterative function
[ibi,indx,ECG,ECG_corr,peak_indx,ADD,REM] = editpeaks(ibi,indx, ECG, ECG_corr,peak_indx,time, fsample,ADD, REM,M);

% visualize edition result
close all
figure
hist(ibi,20)
title('Heart-Rate Histogram')
xlabel('Time [s]')
ylabel('Frequency')

figure
ECGplot1 = ECG_corr*15;
ECGplot1(ECGplot1>5) = 5;
plot(ECGplot1)
hold on
plot(indx,ECGplot1(indx),'ok')
hold on
ECGplot2 = zscore(ECG)+10;
ECGplot2(ECGplot2>15) = 15;
plot(ECGplot2)
hold on
plot(indx,ECGplot2(indx),'ok')
title('EEG signal with ECG artifacts')
xlabel('[samples]')
ylabel('[uV]')
ylim([0 15])
set(gcf,'units','points','position',[10,10,1200,300])
end

