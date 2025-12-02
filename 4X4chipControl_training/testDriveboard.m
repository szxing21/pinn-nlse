clc; clear all;
%%
[a,dac]=InitArduino(1000000); % Arduino board 
ResetDac(a); % Reset DAC board
% Pause time (s)
pause_write = 0.1;
%%

% Apply weights
for i=0:1:39
    Write2dac_test(dac, i,22000,0.1); 
end

%%

Write2dac_test(dac, 18, 60000, pause_write); 