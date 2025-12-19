figure;
tiledlayout(10,10, 'TileSpacing','compact', 'Padding','compact');  % 5×10 = 50

for k = 1:100
    fname = sprintf('./Rx/workspace_mvm_%d.mat', k);
    
    nexttile;
    
    if ~isfile(fname)
        title(sprintf('Run %d (missing)', k));
        axis off;
        continue;
    end
    
    load(fname);   % 需要包含 out_eq 和 idealOut
    
    if ~exist('out_eq','var') || ~exist('idealOut','var')
        title(sprintf('Run %d (invalid)', k));
        axis off;
        clear out_eq idealOut
        continue;
    end
    
    scatter(out_eq, idealOut, 6, '.');
    axis tight;
    grid on;
    title(sprintf('Run %d', k), 'FontSize', 8);
    
    clear out_eq idealOut
end

sgtitle('Measured vs. Ideal Output for Multiple Runs');
