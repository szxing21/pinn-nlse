function [out_eq, out_raw] = run_mvm_eq(W_tile, X_tile)
%HW_BLOCK_MUL  Run a 4x4 hardware MVM and Volterra均衡.
%   W_tile  : 4x4 权重矩阵
%   X_tile  : 4xB 输入样本（每列一个样本）
%   CTool   : ConvTool 实例
%   *_AWG/pd*/laser/dac : 已经 fopen 的仪器句柄
%
%   返回:
%     out_raw - 硬件测量的 4xB 输出
%     out_eq  - 经 votelrra 均衡后的 4xB 输出
    CTool = ConvTool;
    [dac, tekAWG, agAWG, pd1, pd2, laser] = initInstr;
    B_bias = [0;0;0;0];
    [out_raw, ~] = CTool.multiSampMVM_ver2(X_tile, W_tile, B_bias, tekAWG, agAWG, laser, pd1, pd2, dac);

    idealOut = W_tile * X_tile;

    taps_linear = 3;
    taps_quadratic = 1;
    taps_cubic = 1;
    numofTs = size(out_raw, 2);
    out_eq = zeros(size(out_raw));
    for r = 1:size(out_raw,1)
        compensated_col = volterra(taps_linear, taps_quadratic, taps_cubic, numofTs, out_raw(r,:).', idealOut(r,:).');
        out_eq(r,:) = compensated_col(:).';
    end
end

