function[dac, tekAWG, agAWG, pd1, pd2, laser] = initInstr

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

end