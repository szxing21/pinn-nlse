which -all Arduino 

function delayValue = getResetDelayHook(obj) 
% workaround for Arduino Mega on Linux taking longer time to reset, explained in g1638539 
if(obj.ConnectionType==matlabshared.hwsdk.internal.ConnectionTypeEnum.Serial) 
    if ispc || ismac 
        delayValue = 2; 
    else 
        delayValue = 10; 
    end 
else 
    %The reset on the DTR line is via Serial only 
    delayValue = 0; 
end 
end 

a = arduino('COM3','Mega2560','Trace',true);