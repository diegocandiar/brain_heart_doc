function [ibi,indx,ECG,ECGnew,peak_indx,ADD,REM] = editpeaks(ibi,indx,ECG,ECGnew,peak_indx,time,freq,ADD,REM,MM)

% This function does the editing of R-peaks detected using the processin_ecg.m 
% The editing is manually. Follow the instruction that will appear in the
% commmand window.
%
% inputs
% ibi: heart rate (interbeat intervals)
% indx: samples where are located the heartbeats
% ECGnew: struct with the EKG
% peak_indx: logical array with 1 where is located the heartbeat
% time: time array of the whole dataset
% ADD: samples of the peaks added
% REM: samples of the peaks removed
% MM: y-limit for the plot (uV)
%
% outputs
% ibi: heart rate (interbeat intervals)
% indx: samples where are located the heartbeats
% ECGnew: struct with the EKG
% peak_indx: logical array with 1 where is located the heartbeat
% ADD: samples of the peaks added
% REM: samples of the peaks removed
%
% Author: Diego Candia-Rivera 
% diego.candia.r@ug.uchile.cl
% To refer to this code please cite the following publication:
% XXXXXXXXXXXXXXXXXXXXXXXXXXXX

s1 = input('Would you like to modify heartbeats manually (y/n)? ', 's');
    if strcmp(s1,'y')
        minhr = input('Min heart rate in seconds: ');
        maxhr = input('Max heart rate in seconds: ');
    badindx=[];
    for i=1:length(ibi)
        if ibi(i) < minhr || ibi(i) > maxhr
            j = indx(i);
            badindx=[badindx j];
        end
    end
    
    if length(badindx) >= 1
    
        
        
    close all
    
    figure
    ECGplot1 = ECGnew*15;
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
    hold on
    plot(badindx,15,'*c')
    title('EEG signal with ECG artifacts')
    xlabel('[samples]')
    ylabel('[uV]')
    ylim([0 15])
    set(gcf,'units','points','position',[10,10,1200,300])

   
    
    else
        fprintf('There is not unusual heart rates\n')
    end
    
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
                    % plot(ecg);
                    hold on
                    plot(ol,au,'or','markersize',8)
                    if ol - 5 <= 0 % this takes care of the start of the window
                        peak_indx(1:ol+5) = 0;
                    elseif ol + 5 > length(peak_indx) % this takes care of the end of the window
                        peak_indx(ol-5:end) = 0;
                    else
                        peak_indx(ol-5:ol+5) = 0;
                    end

                    % add the desired peak
                elseif strcmp(s3,'a')
                    ADD = [ADD ol];
                    display('Adding peak...')
                    % plot the added peak
                    % plot(ecg);
                    hold on
                    plot(ol,au,'og','markersize',8)
                    if ol - 5 <= 0 % this takes care of the start of the window
                        [~,mx] = max(ECG(1:ol+5));
                        peak_indx(mx) = 1;
                    elseif ol + 5 > length(peak_indx) % this takes care of the end of the window
                        [~,mx] = max(ECG(ol-5:end));
                        peak_indx(mx + ol -5 - 1) = 1;
                    else
                        [~,mx] = max(ECG(ol-5:ol+5));
                        peak_indx(mx + ol -5 - 1) = 1;
                    end
                end
                s3 = input('Would you like to add/remove further peaks (a/r/n) (a = add / r = remove / n = no)? ', 's');
            end
    indx = [];
    for i = 1:length(peak_indx)
        if peak_indx(i) > 0.5 % detect if the index is different to zero
            indx = [indx i];
        end
    end

    close all
    ibi = calc_heartrate(indx,time);
    figure
    hist(ibi,20)
    title('Heart-Rate Histogram')
    xlabel('Time [s]')
    ylabel('Frequency')

    figure
    ECGplot1 = ECGnew*15;
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
    
    [ibi,indx,ECG,ECGnew,peak_indx,ADD,REM] = editpeaks(ibi,indx,ECG,ECGnew,peak_indx,time,freq,ADD,REM,MM);

    end
    

close all