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

%%
out = query(tekAWG, '*IDN?');
disp(out)

%%
fprintf(tekAWG, 'SOUR1:FUNC DC');
fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:HIGH 2.45V');
fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:LOW 0.0V');

fprintf(tekAWG, 'SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(tekAWG, ':OUTP1 ON');

out_power = [];


for i = 0:0.005:4.9
    fprintf(tekAWG, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
%%
fprintf(tekAWG, 'SOUR2:FUNC DC');
fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.45V');
fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');

fprintf(tekAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(tekAWG, ':OUTP2 ON');

out_power = [];


for i = 0:0.005:4.9
    fprintf(tekAWG, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
fprintf(tekAWG, ':OUTP2 OFF');
%%

figure(1)
plot(0:0.005:4.9, out_power)

%%
save("InputCalib\in3caldata.mat", "out_power")

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

out_power = [];


for i = 0:0.005:4.9
    fprintf(agAWG, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
%%
fprintf(agAWG, ':OUTP1 OFF');
%%
save("InputCalib\in4caldata.mat", "out_power")
%%

fprintf(agAWG, 'SOUR2:FUNC DC');
fprintf(agAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.45V');
fprintf(agAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');
fprintf(agAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
fprintf(agAWG, ':OUTP2 ON');

out_power = [];


for i = 0:0.005:4.9
    fprintf(agAWG, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', i/2));
    pause(0.1)
    out_power(end+1) = str2double(query(pd1, 'READ1:POW?'));
end
%%
fprintf(agAWG, ':OUTP2 OFF');