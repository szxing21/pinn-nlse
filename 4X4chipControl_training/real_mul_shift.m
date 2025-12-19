function Y = real_mul_shift(A, B)
%REAL_MUL_SHIFT  有符号实数矩阵乘，通过平移+归一化调用 run_mvm_eq，再数字域补偿。
%   A: m x k 实数，B: k x n 实数
%   步骤：
%     1) 平移到非负 Wp = A + c, Xp = B + d（c,d 取负最小值）
%     2) 归一化到 [0,1]：Wn = Wp/max(Wp), Xn = Xp/max(Xp)
%     3) 调用 run_mvm_eq(Wn, Xn) 计算主项，输出再乘回缩放因子
%     4) 补偿项在数字域：A*B = Wp*Xp - d*sum(Wp,2)*1' - c*1*sum(Xp,1) + k*c*d
    clearvars -except A B
    save('workspace_AB.mat');
    if ~isreal(A) || ~isreal(B)
        error('Inputs must be real.');
    end
    epsc = 0.01*max(max(abs(A)));
    epsd = 0.01*max(max(abs(B)));
    c = -min(A(:)); c = max(c, 0)+epsc;
    d = -min(B(:)); d = max(d, 0)+epsd;
    Wp = A + c;
    Xp = B + d;

    scale_w = max(Wp(:)); if scale_w == 0, scale_w = 1; end
    scale_x = max(Xp(:)); if scale_x == 0, scale_x = 1; end
    Wn = Wp / scale_w;
    Xn = Xp / scale_x;

    % 主项：非负且归一化的矩阵乘，依赖 run_mvm_eq(Wn, Xn)
    Ypos = run_mvm_eq(Wn, Xn);
    % Ypos = Wn*Xn;

    Ypos = Ypos * (scale_w * scale_x);

    m = size(A,1);
    n = size(B,2);
    k = size(A,2);

    sumW = sum(Wp, 2);          % m x 1
    sumX = sum(Xp, 1);          % 1 x n
    corr1 = d * sumW * ones(1, n);        % d * Wp * 1
    corr2 = c * ones(m,1) * sumX;         % c * 1 * Xp
    corr3 = k * c * d;                    % c*d*1*1 (scalar)

    Y = Ypos - corr1 - corr2 + corr3;
    idx = str2double(fileread('index.txt'));
    save(sprintf('./Rx/workspace_real_%d.mat', idx));
    fid = fopen('index.txt','w');
    fprintf(fid,'%d', idx+1);
    fclose(fid);

end
% %% test
% close all;
% clear;
% 
% CTool = ConvTool;
% 
% rng(12)
% Mat1 = rand([4,4])-0.5;
% rng(123)
% X1 = rand([4,50])-0.5;
% idealOut = Mat1 * X1;
% out_eq = real_mul_shift(Mat1, X1);