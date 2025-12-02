function [a,dac] = InitArduino(bitrate)
%ArduinoInit: Initial setup of Arduino board

% bitrate: spi bitrate from arduino board to dac
% a: arduino object
% dac: spi device object

% Create an arduino object and include the SPI library
a = arduino('COM3', 'Mega2560', 'Libraries', 'SPI');
% Create an SPI device object
dac = device(a, 'SPIChipSelectPin', 'D53', 'BitOrder', 'msbfirst', 'SPIMode', 1,'bitrate',bitrate);

% Configure Pins for controlling DAC
configurePin(a,'D28', 'DigitalOutput'); writeDigitalPin(a,'D28',0); % LDAC' low --> all outputs are updated simultaneously
configurePin(a,'D30', 'DigitalOutput'); writeDigitalPin(a,'D30',1); % CLR' high --> can program DAC registers
configurePin(a,'D34', 'DigitalOutput'); writeDigitalPin(a,'D34',0); % RESET' low --> reset all DAC registers

end