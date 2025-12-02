function [] = Write2dac_test(dac,port_number,voltage,pause_write)
%Write2dac Send a voltage data to dac

% obj: oscilloscope object
% dac: dac object
% port_number: dac port number (address)
% voltage: voltage in decimal (dac data register 5554 (21844) to f67c (63100) <--> 0 to 8V dac output)
% pause_write: pause time after write a voltage to DAC

address = 200+port_number; % according to ad5370 datasheet
tmp = dec2hex(voltage);
data = [hex2dec(tmp(1:2)) hex2dec(tmp(3:4))];
dataIn = [address data];
writeRead(dac,dataIn);
pause(pause_write);

end