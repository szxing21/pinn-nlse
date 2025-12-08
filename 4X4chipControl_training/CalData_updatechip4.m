V_range = 22100:100:58000;

%%

dataColCellM = {};
dataColCellR = {};
mat_ele_n = [11, 12, 13, 14; 21, 22, 23, 24; 31, 32, 33, 34; 41, 42, 43, 44];

for i = 1:1:4
    for j = 1:1:4
        readFileNameM = sprintf("chip3Calib/mat_ele_%d_Pmax",mat_ele_n(i,j));
        readFileNameR = sprintf("chip3Calib/mat_ele_%d_Pref",mat_ele_n(i,j));
        load(readFileNameM);
        load(readFileNameR);
        dataColCellM{i,j} = OutPowerVal(2:end);
        dataColCellR{i,j} = OutPowerRef(2:end);
    end
end

%%
j = 4;
figure(1);
plot(V_range, dataColCellM{1,j})
hold on
plot(V_range, dataColCellM{2,j})
hold on
plot(V_range, dataColCellM{3,j})
hold on
plot(V_range, dataColCellM{4,j})

%%
for i = 1:1:4
    for j = 1:1:4
        maxDR(i,j) = min(dataColCellM{i,j}) - max(dataColCellM{i,j});
        dataColPow{i,j} = db2pow(dataColCellM{i,j} - max(dataColCellM{i,j}));
    end
end

smoothPow = {};
for i = 1:1:4
    for j = 1:1:4
        smoothPow{i, j} = smoothdata(dataColPow{i,j}, "movmean", 20);
    end
end

%%
i = 1;
figure(1)
hold on
plot(V_range, smoothPow{i,1})
plot(V_range, smoothPow{i,2})
plot(V_range, smoothPow{i,3})
plot(V_range, smoothPow{i,4})
plot(V_range, dataColPow{i,1})
plot(V_range, dataColPow{i,2})
plot(V_range, dataColPow{i,3})
plot(V_range, dataColPow{i,4})

%%

concList = {};
for i = 1:1:4
    for j = 1:1:4
        [~, minInd] = min(smoothPow{i,j});
        concList{i,j} =smoothPow{i,j}(1:minInd);

    end
end

i = 1;
figure(1)
hold on

plot(concList{i,1})
plot(concList{i,2})
plot(concList{i,3})
plot(concList{i,4})


%%

levelList = 0.01:0.01:1;
mat_ele_n = [11, 12, 13, 14; 21, 22, 23, 24; 31, 32, 33, 34; 41, 42, 43, 44];
for i = 1:1:4
    for j = 1:1:4
        voltRef = [];
        for k = 1:1:100
            [~, IndSel] = min(abs(levelList(k)-concList{i,j}));
            voltRef(k) = V_range(IndSel);
        end
        conListName = sprintf("RefConst4/Element_%d_conList.mat", mat_ele_n(i,j));
        save(conListName, "voltRef")
    end
end



