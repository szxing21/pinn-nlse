function [] = ResetDac(a)
%DacReset Reset DAC outputs

% a: arduino object

writeDigitalPin(a,'D34',0);
writeDigitalPin(a,'D34',1);

end