clc; clear; close all;

% 参数设置
L = 100*1e3;                % 总长度 (km)
dz = 100;                % SSFM 步长 = 100 m
Nz = L / dz;        % 总步数

beta2 = -21.242e-27;    % 蓝线 beta2
beta3 = 16.6e-41;       % 蓝线 beta3
gamma = 1.3;           
gamma_SI = gamma / 1e3; % (1/W/m)

T = 3000e-12;           
N = 2^10;               % 采样点
dt = T / N;
t = (-N/2:N/2-1)*dt;    % 时间序列

P0 = 0.5;                 % 峰值功率
FWHM = 100e-12;          
sigma = FWHM / (2*sqrt(2*log(2)));
delays = [-400 0 400]*1e-12;   % 三个脉冲
A0 = zeros(size(t));         % 初始化脉冲
for k = 1:length(delays)
    d = delays(k);
    A0 = A0 + sqrt(P0) * exp(-(t - d).^2 / (2*sigma^2));
end
A0 = A0.';
% A0 = sqrt(P0) * exp(-t.^2/(2*sigma^2)).';   % 初始高斯脉冲
df = 1/(N*dt);
f = (-N/2:N/2-1)*df;
omega = 2*pi*f;

% -----------------------
% 保存矩阵设置
% -----------------------
save_interval = 10e3;              % 每 1 km 保存一次 (m)
num_save = L/save_interval;                     % 总共保存100个点 (1-100 km)，加上0一共101
Tensor = zeros(num_save+1, N, 2); % (z,t,Re/Im)
Tensor(1,:,1) = real(A0);
Tensor(1,:,2) = imag(A0);

A = A0;
counter = 1;                      % 保存计数
z_current = 0;                    % 当前距离 (m)

% -----------------------
% SSFM 演化
% -----------------------
for step = 1:Nz
    % 频域色散
    A_freq = fftshift(fft(A));
    H = exp(-1i*(beta2/2*omega.^2 + beta3/6*omega.^3)*dz);
    A_freq = A_freq .* H.';
    A = ifft(ifftshift(A_freq));

    % 时域非线性
    A = A .* exp(1i*gamma_SI*abs(A).^2*dz);

    % 更新距离
    z_current = z_current + dz;

    % 每1 km保存一次
    if mod(z_current, save_interval) == 0
        counter = counter + 1;
        Tensor(counter,:,1) = real(A);
        Tensor(counter,:,2) = imag(A);
    end
end

% -----------------------
%% 绘制三维瀑布图 (z-t 演化, 独立曲线)
% -----------------------
Z = 0:save_interval:L;             % 0,1,...,100 km
T_ps = t*1e12;       % ps
Power = squeeze(Tensor(:,:,1)).^2 + squeeze(Tensor(:,:,2)).^2;

figure; hold on;
for k = 1:length(Z)
    % 沿 t 方向是横坐标，沿 z 方向推进
    plot3(T_ps, Z(k)*ones(size(T_ps)), Power(k,:), ...
        'b', 'LineWidth', 1.2);  
end
xlabel('Time (ps)');
ylabel('Distance (km)');
zlabel('Power (W)');
title('Pulse evolution along fiber (blue case)');
grid on; view(3);

% 调节时间显示范围
xlim([min(t) max(t)]*1e12);
pbaspect([1 3 1]);    % x : y : z = 1 : 1 : 2
% 设置视角 (方位角=65.25°, 仰角=13.5784°)
view([65.25 13.5784]);

% 设置投影方式
camproj('orthographic');   % 平行投影
% camproj('perspective');  % 透视投影（可选）

% 设置相机位置
campos([37452, -165820, 5.4802]);

% 设置相机目标点
camtarget([-0.0153, 50000, 0.5]);

% 设置相机朝向 (这里保持z轴向上)
camup([0, 0, 1]);



%%
% -----------------------
% 初始波形 vs 最终波形 对比
% -----------------------
A_initial = Tensor(1,:,1) + 1i*Tensor(1,:,2);
A_final   = Tensor(end,:,1) + 1i*Tensor(end,:,2);

figure;
plot(T_ps, abs(A_initial).^2, 'b-', 'LineWidth', 2, 'DisplayName', 'Initial');
hold on;
plot(T_ps, abs(A_final).^2, 'r--', 'LineWidth', 2, 'DisplayName', 'Final');
xlabel('Time (ps)');
ylabel('Power (W)');
title('Initial vs Final Pulse');
legend('show');
grid on;
xlim([min(t) max(t)]*1e12);

%% -----------------------
% 保存结果到 .mat 文件
% -----------------------
Z = 0:save_interval:L;   % z 轴 (m)
T_ps = t;           % 时间轴 (ps)

save('.\pinn-nlse-main\data\pulse_evolution.mat', 'Tensor', 'Z', 'T_ps');
