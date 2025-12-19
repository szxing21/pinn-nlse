function[dac, tekAWG, agAWG, pd1, pd2, laser] = initInstr
objs = instrfind('Type','gpib');
for k = 1:numel(objs)
    try, fclose(objs(k)); end
    try, delete(objs(k)); end
end
clear objs

tekAWG = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0699::0x034C::C010870::0::INSTR', 'Tag', '');
if isempty(tekAWG)
    tekAWG = visa('NI', 'USB0::0x0699::0x034C::C010870::0::INSTR');
else
    fclose(tekAWG);
    tekAWG = tekAWG(1);
end
fopen(tekAWG);

agAWG = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0957::0x2607::MY52200188::0::INSTR', 'Tag', '');
if isempty(agAWG)
    agAWG = visa('NI', 'USB0::0x0957::0x2607::MY52200188::0::INSTR');
else
    fclose(agAWG);
    agAWG = agAWG(1);
end
fopen(agAWG);

% PDs via VISA (visadev)
pd1 = visadev("GPIB0::23::INSTR");
pd2 = visadev("GPIB0::22::INSTR");
% pd1.Timeout = 5; pd2.Timeout = 5;
% pd1.Terminator = "LF"; pd2.Terminator = "LF";

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

end
