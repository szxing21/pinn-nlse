function [Y, outValCol] = run_hw_mvm(X, W, B)
%RUN_HW_MVM  Wrapper to execute a hardware 4x4 matrix-vector multiplication.
%   [Y, outValCol] = RUN_HW_MVM(X, W, B) initialises all instruments,
%   applies the 4x4 weight matrix W and input matrix X (4 x N),
%   and returns the measured output Y along with per-column raw measurements
%   outValCol. Bias B (4 x 1) is optional; if omitted, zeros are used.
%
%   This uses ConvTool.multiSampMVM_ver2 and handles instrument init.
%
%   Example:
%       rng(0); W = rand(4,4); X = rand(4,10); B = zeros(4,1);
%       [Y, rawCols] = run_hw_mvm(X, W, B);

    if nargin < 3
        B = zeros(4,1);
    end

    [dac, tekAWG, agAWG, pd1, pd2, laser] = initInstr;

    CTool = ConvTool;
    [Y, outValCol] = CTool.multiSampMVM_ver2(X, W, B, tekAWG, agAWG, laser, pd1, pd2, dac);
end
