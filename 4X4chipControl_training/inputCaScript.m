clc, clear all;
instrreset;
%%
tekAWG = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0699::0x034C::C010870::0::INSTR', 'Tag', '');
if isempty(tekAWG)
    tekAWG = visa('NI', 'USB0::0x0699::0x034C::C010870::0::INSTR');
else
    fclose(tekAWG);
    tekAWG = tekAWG(1);
end
fopen(tekAWG);

pd1 = instrfind('Type', 'gpib', 'BoardIndex', 0, 'PrimaryAddress', 23, 'Tag', '');
if isempty(pd1)
    pd1 = gpib('NI',0, 23);
else
    fclose(pd1);
    pd1 = pd1(1);
end
fopen(pd1);

laser = instrfind('Type', 'gpib', 'BoardIndex', 0, 'PrimaryAddress', 3, 'Tag', '');
if isempty(laser)
    laser = gpib('NI',0, 3);
else
    fclose(laser);
    laser = laser(1);
end
fopen(laser);
%%
out = query(tekAWG, '*IDN?');
disp(out)

%%
fprintf(tekAWG, 'SOUR1:FUNC DC');
fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:HIGH 2.45V');
fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:LOW 0.0V');

fprintf(tekAWG, 'SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(tekAWG, ':OUTP1 ON');
fprintf(laser, 'CHAN 1; OUT 1');
fprintf(laser, "CHAN 1; SHUTTER 1");
out_power = [];

for i = 0:0.005:4.9
    fprintf(tekAWG, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
fprintf(laser, 'CHAN 1; OUT 0');
fprintf(laser, "CHAN 1; SHUTTER 0");
save("InputCalib\in1caldata.mat", "out_power")
%%
fprintf(tekAWG, 'SOUR2:FUNC DC');
fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.45V');
fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');

fprintf(tekAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(tekAWG, ':OUTP2 ON');
fprintf(laser, 'CHAN 2; OUT 1');
fprintf(laser, "CHAN 2; SHUTTER 1");
out_power = [];


for i = 0:0.005:4.9
    fprintf(tekAWG, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
fprintf(tekAWG, ':OUTP2 OFF');

% figure(1)
% plot(0:0.005:4.9, out_power)
fprintf(laser, 'CHAN 2; OUT 0');
fprintf(laser, "CHAN 2; SHUTTER 0");
save("InputCalib\in2caldata.mat", "out_power")

%%

agAWG = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0957::0x2607::MY52200188::0::INSTR', 'Tag', '');
if isempty(agAWG)
    agAWG = visa('NI', 'USB0::0x0957::0x2607::MY52200188::0::INSTR');
else
    fclose(agAWG);
    agAWG = agAWG(1);
end
fopen(agAWG);

%%

fprintf(agAWG, 'SOUR1:FUNC DC');
fprintf(agAWG, 'SOURce1:VOLTage:LIMit:HIGH 2.45V');
fprintf(agAWG, 'SOURce1:VOLTage:LIMit:LOW 0.0V');
fprintf(agAWG, 'SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(agAWG, ':OUTP1 ON');
fprintf(laser, 'CHAN 3; OUT 1');
fprintf(laser, "CHAN 3; SHUTTER 1");
out_power = [];


for i = 0:0.005:4.9
    fprintf(agAWG, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
fprintf(agAWG, ':OUTP1 OFF');
fprintf(laser, 'CHAN 3; OUT 0');
fprintf(laser, "CHAN 3; SHUTTER 0");
save("InputCalib\in3caldata.mat", "out_power")
%%

fprintf(agAWG, 'SOUR2:FUNC DC');
fprintf(agAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.45V');
fprintf(agAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');
fprintf(agAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(agAWG, ':OUTP2 ON');
fprintf(laser, 'CHAN 4; OUT 1');
fprintf(laser, "CHAN 4; SHUTTER 1");
out_power = [];


for i = 0:0.005:4.9
    fprintf(agAWG, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
fprintf(laser, 'CHAN 4; OUT 0');
fprintf(laser, "CHAN 4; SHUTTER 0");
save("InputCalib\in4caldata.mat", "out_power")
fprintf(agAWG, ':OUTP2 OFF');