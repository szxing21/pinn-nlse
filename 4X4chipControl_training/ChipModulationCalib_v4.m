clc, clear all;
instrreset;

%%

pd1 = instrfind('Type', 'gpib', 'BoardIndex', 0, 'PrimaryAddress', 23, 'Tag', '');
if isempty(pd1)
    pd1 = gpib('NI',0, 23);
else
    fclose(pd1);
    pd1 = pd1(1);
end
fopen(pd1);

pd2 = instrfind('Type', 'gpib', 'BoardIndex', 0, 'PrimaryAddress', 22, 'Tag', '');
if isempty(pd2)
    pd2 = gpib('NI',0, 22);
else
    fclose(pd2);
    pd2 = pd2(1);
end
fopen(pd2);
laser = instrfind('Type', 'gpib', 'BoardIndex', 0, 'PrimaryAddress', 3, 'Tag', '');
if isempty(laser)
    laser = gpib('NI',0, 3);
else
    fclose(laser);
    laser = laser(1);
end
fopen(laser);
[a,dac]=InitArduino(1000000); % Arduino board 
ResetDac(a); % Reset DAC board
pause_write = 0.2;
VoltHexRange = [21844, 55300];
disp(1)


%%

idn2_pd1 = query(pd1, '*IDN?');
idn3_pd2 = query(pd2, '*IDN?');
disp(idn2_pd1)
disp(idn3_pd2)

%%

fprintf(pd1, 'SENS1:POW:ATIME 50MS');
fprintf(pd1, 'SENS2:POW:ATIME 50MS');
%%
fprintf(pd2, 'SENS1:POW:ATIME 50MS');
fprintf(pd2, 'SENS2:POW:ATIME 50MS');
%%
fprintf(pd2, 'SENS2:POW:RANG -20DBM');
fprintf(pd1, 'SENS1:POW:RANG -20DBM');
fprintf(pd1, 'SENS2:POW:RANG -20DBM');


%%

mat_ele_n = [11, 12, 13, 14; 21, 22, 23, 24; 31, 32, 33, 34; 41, 42, 43, 44];
mat_chan_list = [5, 4, 7, 12; 3, 6, 11, 16; 25, 30, 33, 38; 27, 26, 31, 34];
pd_items = {pd1, pd2};

in_start = 1;
in_end = 1;
out_start = 2;
out_end = 2;
V_range = 22000:100:58000;
pause_write = 0.1;

dcCol = [];
dc2Col = [];

for i = in_start:1:in_end
    fprintf(laser, 'CHAN %d; OUT 1', i);
    fprintf(laser, 'CHAN %d; SHUTTER 1', i);
    for j = out_start:1:out_end
        OutPowerVal = [];
        OutPowerRef = [];
        if j == 1
            pdChanSelRead = sprintf('READ%d:POW?', int8(1));
            pdChanSelRefe = sprintf('READ%d:POW?', int8(2));
            for k = 22000:100:58000
                Write2dac_test(dac, mat_chan_list(j,i), k, pause_write);
                pause(0.2)
                pdOutSingle = str2double(query(pd1, pdChanSelRead));
                pdOutRefSin = str2double(query(pd1, pdChanSelRefe));
                OutPowerVal(end+1) = pdOutSingle;
                OutPowerRef(end+1) = pdOutRefSin;
            end
        elseif j == 2
            pdChanSelRead = sprintf('READ%d:POW?', int8(2));
            pdChanSelRefe = sprintf('READ%d:POW?', int8(1));
            for k = 22000:100:58000
                Write2dac_test(dac, mat_chan_list(j,i), k, pause_write);
                pause(0.2)
                pdOutSingle = str2double(query(pd1, pdChanSelRead));
                pdOutRefSin = str2double(query(pd1, pdChanSelRefe));
                OutPowerVal(end+1) = pdOutSingle;
                OutPowerRef(end+1) = pdOutRefSin;
            end
        elseif j == 3
            pdChanSelRead = sprintf('READ%d:POW?', int8(1));
            pdChanSelRefe = sprintf('READ%d:POW?', int8(2));
            for k = 22000:100:58000
                Write2dac_test(dac, mat_chan_list(j,i), k, pause_write);
                pause(0.2)
                pdOutSingle = str2double(query(pd2, pdChanSelRead));
                pdOutRefSin = str2double(query(pd2, pdChanSelRefe));
                OutPowerVal(end+1) = pdOutSingle;
                OutPowerRef(end+1) = pdOutRefSin;
            end
        elseif j == 4
            pdChanSelRead = sprintf('READ%d:POW?', int8(2));
            pdChanSelRefe = sprintf('READ%d:POW?', int8(1));
            for k = 22000:100:58000
                Write2dac_test(dac, mat_chan_list(j,i), k, pause_write);
                pause(0.2)
                pdOutSingle = str2double(query(pd2, pdChanSelRead));
                pdOutRefSin = str2double(query(pd2, pdChanSelRefe));
                OutPowerVal(end+1) = pdOutSingle;
                OutPowerRef(end+1) = pdOutRefSin;
            end
        end
        Write2dac_test(dac, mat_chan_list(j,i), 21900, pause_write);
        f_namePow = sprintf("chip3Calib/mat_ele_%d_Pmax.mat", mat_ele_n(j,i));
        f_nameRef = sprintf("chip3Calib/mat_ele_%d_Pref.mat", mat_ele_n(j,i));
        save(f_namePow, "OutPowerVal");
        save(f_nameRef, "OutPowerRef");
        disp('Out * 1')
        figure(1)
        plot(V_range(2:end), OutPowerVal(2:end))
        hold on
    end
    fprintf(laser, 'CHAN %d; OUT 0', i);
    fprintf(laser, 'CHAN %d; SHUTTER 0', i);
end

%%

% plot( OutPowerVal(2:end))
% 
% %%
% PMax11 = [];
% load("data/mat_ele_11_Pmax.mat");
% load("data/mat_ele_11_Pref.mat");
% disp(1)
% 
% V_range = 21900:100:55300;
% %plot(V_range, OutPowerRef)
% 
% Rev_change = [];
% size2 = size(OutPowerRef);
% for i = 1:1:(size2(2));
%     Rev_change(end+1) = OutPowerRef(i) - OutPowerRef(1);
% end
% 
% OutPowerCal = OutPowerVal - Rev_change;
% figure(1);
% plot(V_range, OutPowerVal)
% figure(2);
% plot(V_range, OutPowerCal)
% 
% 
% %%
% port_number = 24;
% pause_write = 0.1;
% Dac_data = 22000:100:60000;
% 
% 
% P_sweep_ch24_1 = [];
% Vheater_tune_ch24_1 = [];
% Iheater_tune_ch24_1 = [];
% 
% 
% for i=1:1:length(Dac_data)
%     Write2dac(dac,port_number,Dac_data(i),pause_write);  % write a voltage to dac
%     Vheater_tune_ch24_1 = [Vheater_tune_ch24_1; str2num(query(MMV, ':MEAS:VOLT:DC?'))];
%     Iheater_tune_ch24_1 = [Iheater_tune_ch24_1; str2num(query(MMC, ':MEAS:CURR:DC?'))];
%     P_sweep_ch24_1 = [P_sweep_ch24_1 str2num(query(PD, 'READ2:POW?'))]; %READ1:chA; READ2:chB
% end
% 
% figure;plot(Vheater_tune_ch24_1,P_sweep_ch24_1);
% %%
% plot(V_range, dc2Col)
