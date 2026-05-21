%% ============= 全局关闭记录以节省磁盘空间 =============
model_name = 'LPM_model';  % Simulink 模型名（不带 .slx）
load_system(model_name);
set_param(model_name, 'SignalLogging', 'off');
set_param(model_name, 'DSMLogging', 'off');
set_param(model_name, 'ReturnWorkspaceOutputs', 'on');

%% ============= 配置区（请根据需要微调） =============
sim_time   = 60;                    % 每次仿真时长（秒）
N_samples  = 1000;                   % 想要的样本总数，测试通过后可增加
heart_rate = 1/0.8;                 % 心率（Hz），对应 T=0.8s
period     = 1/heart_rate;          % 心动周期长度
last_n_cycles = 3;                  % 提取最后 N 个稳态周期

% ---- 需要变化的参数及其采样范围 ----
param_fields = {'Elvmax', 'Ervmax', 'R_sar', 'C_ao', 'TBV'};
param_range = [
    1.5, 4.0;    % Elvmax
    0.6, 2.0;    % Ervmax
    0.5, 2.5;    % R_sar
    0.04, 0.15;  % C_ao
    700, 1200    % TBV
];
n_params = length(param_fields);

% ---- 默认基础参数（直接复制你手动输入的结构体） ----
base_params_default = struct('R_ao',0.003,'C_ao',0.08,'L_ao',0.0000168,...
                     'R_sat', 0.05, 'L_sat', 0.0017, 'C_sat',1,...
                     'R_sar', 0.5+0.52, ...
                     'R_svn', 0.075,'C_svn',20.5,...
                     'Ervmax',1.15, 'T',0.8,'Elvmax',2.5,'TBV',950, ...
                     'R_pas',0.002,'C_pas',0.18, 'L_pas',0.000052, ...
                     'R_pat',0.01+0.04,'C_pat',3.8, 'L_pat',0.00017, ... 
                     'R_pcp',0.05, ...
                     'R_pvn',0.006,'C_pvn',20.5);

%% ============= 生成样本（拉丁超立方或均匀随机） =============
if exist('lhsdesign', 'file')
    fprintf('使用拉丁超立方采样...\n');
    sample = lhsdesign(N_samples, n_params);
else
    fprintf('未安装统计工具箱，使用标准均匀随机采样作为替代...\n');
    rng(42);
    sample = rand(N_samples, n_params);
end

% 缩放至各参数的实际范围
params = zeros(N_samples, n_params);
for j = 1:n_params
    params(:,j) = param_range(j,1) + (param_range(j,2)-param_range(j,1)) * sample(:,j);
end

%% ============= 初始化存储 =============
n_features = 6;  % LVEDV, LVESV, LVEF, RVEDV, RVESV, MAP
X = zeros(N_samples, n_features);
Y = zeros(N_samples, n_params);
success = false(N_samples, 1);

%% ============= 主循环 =============
fprintf('开始批量仿真，共计 %d 组参数...\n', N_samples);
tic;

for i = 1:N_samples
    % 实时进度显示（每 10 次打印一次，省得刷屏）
    if mod(i, 10) == 0
        fprintf('进度：%d/%d\n', i, N_samples);
    end
    
    % ----- 1. 构建本次的 base_params 并推入基础工作区 -----
    base_params = base_params_default;
    for j = 1:n_params
        base_params.(param_fields{j}) = params(i, j);
    end
    assignin('base', 'base_params', base_params);
    
    % ----- 2. 运行仿真 -----
    try
        simOut = sim(model_name, 'StopTime', num2str(sim_time), ...
                     'SrcWorkspace', 'base', ...
                     'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
                     'SignalLogging', 'off', ...
                     'DSMLogging', 'off', ...
                     'ReturnWorkspaceOutputs', 'on');
    catch ME
        warning('组合 %d 仿真启动失败：%s', i, ME.message);
        continue;
    end
    
        % ----- 3. 信号提取（多重保护） -----
    try
        t = simOut.get('tout');
        lv_vol  = simOut.get('LVvolume').Data(:);
        rv_vol  = simOut.get('RVvolume').Data(:);
        ao_pres = simOut.get('AP').Data(:);
        
        % 只检查信号长度是否与时间向量一致
        if length(lv_vol) ~= length(t)
            warning('第 %d 次仿真：信号长度与时间不匹配，跳过。', i);
            continue;
        end
    catch ME
        warning('第 %d 次信号提取失败：%s', i, ME.message);
        continue;
    end
    
    % ----- 4. 稳态区间提取（带边界保护） -----
    try
        t_end = t(end);
        t_steady_start = t_end - last_n_cycles * period;
        idx_steady = find(t >= t_steady_start);
        idx_steady(idx_steady > length(t)) = [];
        
        if length(idx_steady) < 10   % 稳态点太少，说明异常
            warning('第 %d 次：稳态点数不足 %d，跳过。', i, length(idx_steady));
            continue;
        end
        
        lv_vol_steady  = lv_vol(idx_steady);
        rv_vol_steady  = rv_vol(idx_steady);
        ao_pres_steady = ao_pres(idx_steady);
        
        LVEDV = double(max(lv_vol_steady));
        LVESV = double(min(lv_vol_steady));
        RVEDV = double(max(rv_vol_steady));
        RVESV = double(min(rv_vol_steady));
        LVEF  = (LVEDV - LVESV) / LVEDV;
        MAP   = double(mean(ao_pres_steady));
        
    catch ME
        warning('第 %d 次稳态计算失败：%s', i, ME.message);
        continue;
    end
    
    % ----- 5. 生理范围过滤 -----
    if any(LVEDV(:) <= 0) || any(LVEDV(:) > 500) || ...
       any(LVEF(:) < 0.1) || any(LVEF(:) > 0.9) || ...
       any(MAP(:) > 200) || any(MAP(:) < 30)
        warning('第 %d 次结果非生理（LVEDV=%.1f, EF=%.2f, MAP=%.1f），舍弃。', ...
                i, LVEDV, LVEF, MAP);
        continue;
    end
    
    % ----- 6. 存储结果 -----
    X(i, :) = [LVEDV, LVESV, LVEF, RVEDV, RVESV, MAP];
    Y(i, :) = params(i, :);
    success(i) = true;
end

% ----- 清理 -----
close_system(model_name, 0);
total_time = toc;
fprintf('批量仿真完成，总用时 %.1f 秒。\n', total_time);

%% ============= 后处理 =============
X_clean = X(success, :);
Y_clean = Y(success, :);
fprintf('成功样本数：%d / %d，失败/舍弃：%d\n', sum(success), N_samples, N_samples - sum(success));

if sum(success) == 0
    error('没有任何成功样本！请检查模型与信号名称。');
end

%% ============= 保存数据集（指定保存路径） =============
% 你可以修改这里为任意你想保存的完整路径，例如：
% save_path = 'D:\LPM_data\synthetic_dataset.mat';
save_path = fullfile(pwd, 'synthetic_dataset.mat');  % 默认保存在当前工作目录
save(save_path, 'X_clean', 'Y_clean', 'param_fields', 'param_range');
fprintf('数据集已保存至: %s\n', save_path);