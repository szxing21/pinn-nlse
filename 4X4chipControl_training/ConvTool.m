classdef ConvTool < handle
    properties(Constant)
        mat_ele_n = [11, 12, 13, 14; 21, 22, 23, 24; 31, 32, 33, 34; 41, 42, 43, 44];
        discreLev = 0.01:0.01:1;
        mat_chan_list = [5, 4, 7, 12; 3, 6, 11, 16; 25, 30, 33, 38; 27, 26, 31, 34];
        ActLev = 0.01:0.01:1;
        atten_ports = [1, 2, 5, 6];
        chan_OFFS = [-0.49, 0, -0.38, -0.33];
        awgVoltRange = 0:0.005:4.9;
        dac_vrange = 22100:100:58000;
    end

    methods(Static)
        function db_set = NormVal2dB(x)
            set = ConvTool.set2ClosestAct(x);
            db_set = pow2db(set);
        end

        function in = clipArray(in, min, max)
            in(in>max) = max;
            in(in<min) = min;
        end

        function VecOrMat = set2ClosestAct(VecOrMat)
            InShape = size(VecOrMat);
            i_ran = InShape(1);
            j_ran = InShape(2);
            for i = 1:1:i_ran
                for j = 1:1:j_ran
                    [~, SelInd] = min(abs(VecOrMat(i,j)-ConvTool.ActLev));
                    VecOrMat(i,j) = ConvTool.ActLev(SelInd);
                end
            end
        end

        function SetAttVal = set2attendB(InVec)
            SetAttVal = pow2db(InVec);
        end

        function VHexVal = Trans2V(setTransMat, VHexRef, i_ran, j_ran)
            for i = 1:i_ran
                for j = 1:j_ran
                    IndSel = find(ConvTool.ActLev == setTransMat(i,j));
                    VHexVal(i,j) = VHexRef{i,j}(IndSel);
                end
            end
        end

        function PowOut = readPowOut(pdit1, pdit2)
            PowOut = [];
            PowOut(1,1) = str2double(query(pdit1, 'READ1:POW?')); %Output channel 1
            PowOut(2,1) = str2double(query(pdit1, 'READ2:POW?')); %Output channel 2
            PowOut(3,1) = str2double(query(pdit2, 'READ1:POW?')); %Output channel 3
            PowOut(4,1) = str2double(query(pdit2, 'READ2:POW?')); %Output channel 4
        end

        function TtoHexVCol = load_VtoTransCali(i_ran, j_ran) % Function for loading the references of the on chip MZIs
            TtoHexVCol = {};
            for i = 1:1:i_ran
                for j = 1:1:j_ran
                    readFileN = sprintf("RefConst4/Element_%d_conList",ConvTool.mat_ele_n(i,j));
                    load(readFileN);
                    TtoHexVCol{i,j} = voltRef;
                end
            end
        end

        function InCalRef = load_InVcal(MaxRef)
            InCalRef = {};
            for i = 1:1:4
                readFileN = sprintf("InputCalib/in%dcaldata.mat", i);
                load(readFileN);
                InCalRef{i,1} = out_power-MaxRef;
            end
        end

        function InCalRef = load_InVcal2()
            InCalRef = {};
            for i = 1:1:4
                readFileN = sprintf("InputCalib/in%dcaldata.mat", i);
                load(readFileN);
                InCalRef{i,1} = out_power-out_power(10);
            end
        end

        function valVInc = valInAWG(x, incalref)
            valVInc = [];
            x_size = size(x);
            x_db = pow2db(x);
            if x_size(1) ~= 4
                error("Not suitable dimension for the setup")
            end
            for i = 1:1:4
                ref_in = incalref{i,1};
                [~, sel_index] = min(abs(x_db(i,1)-ref_in));
                valVInc(i,1) = ConvTool.awgVoltRange(sel_index);
            end
        end

        function valVInc = valInAWG2(x, incalref)
            valVInc = [];
            x_size = size(x);
            x_db = pow2db(x);
            if x_size(1) ~= 4
                error("Not suitable dimension for the setup")
            end
            for j = 1:1:x_size(2)
                for i = 1:1:4
                    ref_in = incalref{i,1};
                    [~, sel_index] = min(abs(x_db(i,j)-ref_in));
                    valVInc(i,j) = ConvTool.awgVoltRange(sel_index);
                end
            end
        end

        function init_awg(tekAWG, agAWG)
            fprintf(tekAWG, 'SOUR1:FUNC DC');
            fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:HIGH 2.47V');
            fprintf(tekAWG, 'SOURce1:VOLTage:LIMit:LOW 0.0V');
            fprintf(tekAWG, 'SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV');
            fprintf(tekAWG, ':OUTP1 ON');
            fprintf(tekAWG, 'SOUR2:FUNC DC');
            fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.47V');
            fprintf(tekAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');
            fprintf(tekAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
            fprintf(tekAWG, ':OUTP2 ON');
            fprintf(agAWG, 'SOUR1:FUNC DC');
            fprintf(agAWG, 'SOURce1:VOLTage:LIMit:HIGH 2.47V');
            fprintf(agAWG, 'SOURce1:VOLTage:LIMit:LOW 0.0V');
            fprintf(agAWG, 'SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV');
            fprintf(agAWG, ':OUTP1 ON');
            fprintf(agAWG, 'SOUR2:FUNC DC');
            fprintf(agAWG, 'SOURce2:VOLTage:LIMit:HIGH 2.47V');
            fprintf(agAWG, 'SOURce2:VOLTage:LIMit:LOW 0.0V');
            fprintf(agAWG, 'SOURce2:VOLTage:LEVel:IMMediate:OFFSet 0mV');
            fprintf(agAWG, ':OUTP2 ON');
        end

        function closeAWGoutput(tekAWG, agAWG)
            fprintf(tekAWG, ':OUTP1 OFF');
            fprintf(tekAWG, ':OUTP2 OFF');
            fprintf(agAWG, ':OUTP1 OFF');
            fprintf(agAWG, ':OUTP2 OFF');
        end

        function write2InAWG(awg1, awg2, valVIncSin)
            fprintf(awg1, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', valVIncSin(1,1)/2));
            fprintf(awg1, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', valVIncSin(2,1)/2));
            fprintf(awg2, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', valVIncSin(3,1)/2));
            fprintf(awg2, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', valVIncSin(4,1)/2));
        end

        function write2InAWG_single(awg1, awg2, valVIncSin, in_num, sam_num)
            if in_num == 1
                fprintf(awg1, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', valVIncSin(1,sam_num)/2));
            elseif in_num == 2
                fprintf(awg1, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', valVIncSin(2,sam_num)/2));
            elseif in_num == 3
                fprintf(awg2, sprintf('SOUR1:VOLT:LEV:IMM:OFFS %fV', valVIncSin(3,sam_num)/2));
            elseif in_num == 4
                fprintf(awg2, sprintf('SOUR2:VOLT:LEV:IMM:OFFS %fV', valVIncSin(4,sam_num)/2));
            end
        end

        function write2DACmat(dac_item, mat_size, VhexSet)
            pause_write = 0.3;
            for i = 1:1:mat_size(1)
                for j = 1:1:mat_size(2)
                    Write2dac_test(dac_item, ConvTool.mat_chan_list(i,j), VhexSet(i,j), pause_write);
                end
            end
        end

        function write2DACmatColumn(dac_item, mat, column_num)
            pause_write = 0.2;
            vRef = ConvTool.load_VtoTransCali(4, 4);
            [matSize1, ~] = size(mat);
            for i = 1:1:matSize1
                [~, indSel] = min(abs(mat(i,column_num)-ConvTool.ActLev));
                vSet = vRef{i,column_num}(indSel);
                Write2dac_test(dac_item, ConvTool.mat_chan_list(i,column_num), vSet, pause_write);
            end
        end

        function write2DACresetV(dac_item, column_num)
            for i = 1:1:4
                Write2dac_test(dac_item, ConvTool.mat_chan_list(i,column_num), 21800, 0.1)
            end
        end

        function openLaser(laser)
            fprintf(laser, 'CHAN 1; OUT 1');
            fprintf(laser, 'CHAN 2; OUT 1');
            fprintf(laser, 'CHAN 3; OUT 1');
            fprintf(laser, 'CHAN 4; OUT 1');
            fprintf(laser, "CHAN 1; SHUTTER 0");
            fprintf(laser, "CHAN 2; SHUTTER 0");
            fprintf(laser, "CHAN 3; SHUTTER 0");
            fprintf(laser, "CHAN 4; SHUTTER 0");
        end

        function closeLaser(laser)
            fprintf(laser, 'CHAN 1; OUT 0');
            fprintf(laser, 'CHAN 2; OUT 0');
            fprintf(laser, 'CHAN 3; OUT 0');
            fprintf(laser, 'CHAN 4; OUT 0');
        end

    end

    methods
        function [outMat, outValCol] = multiSampMVM_ver2(obj, X, W_M, B, awg1, awg2, laser, pd1, pd2, dac)
            [SampSize1, SampSize2] = size(X);
            [MatSize1, MatSize2] = size(W_M);
            if SampSize1 ~= MatSize2
                error("Matrix and vector size not compatible")
            end
            %outPowCol = {};
            outValCol = {};
            noOfSampforRef = 50;
            All1sMat = ones([4, 4]);
            All1sVec = ones([4, 1]);
            InCalRef = obj.load_InVcal2();
            valVInc = obj.valInAWG2(X, InCalRef);
            valVInc1s = obj.valInAWG(All1sVec, InCalRef);
            obj.openLaser(laser)
            obj.init_awg(awg1, awg2);
            for j = 1:1:SampSize1
                laserCmd1 = sprintf("CHAN %d; SHUTTER 1",j);
                fprintf(laser, laserCmd1);
                outPowTemp = [];
                outPowRef = [];
                outValTemp = [];
                obj.write2DACmatColumn(dac, W_M, j)
                for i = 1:1:SampSize2
                    obj.write2InAWG_single(awg1, awg2, valVInc, j, i)
                    pause(0.2);
                    temp1(:,1) = obj.readPowOut(pd1, pd2);
                    outPowTemp(:,i) = temp1(1:MatSize1,1);
                    if floor(i/noOfSampforRef) == i/noOfSampforRef
                        obj.write2DACmatColumn(dac, All1sMat, j)
                        obj.write2InAWG_single(awg1, awg2, valVInc1s, j, 1)
                        pause(0.2);
                        temp2(:, 1) = obj.readPowOut(pd1,pd2);
                        outPowRef(:, int8(i/noOfSampforRef)) = temp2(1:MatSize1, 1);
                        obj.write2DACmatColumn(dac, W_M, j)
                    end
                end
                obj.write2DACresetV(dac, j);
                [~, noOfRef] = size(outPowRef);
                for k = 1:1:noOfRef
                    for l = 1:1:noOfSampforRef
                        outValTemp(:,(k-1)*noOfSampforRef+l) = db2pow(outPowTemp(:,(k-1)*noOfSampforRef+l)-outPowRef(:,k)) ;
                    end
                end
                %disp(size(outValTemp))
                outValCol{j} = outValTemp;
                laserCmd2 = sprintf("CHAN %d; SHUTTER 0",j);
                fprintf(laser, laserCmd2);
            end
            outMat = outValCol{1};
            for o = 2:1:SampSize1
                outMat = outMat + outValCol{o};
            end
            outMat = outMat + B;
            obj.closeLaser(laser);
            obj.closeAWGoutput(awg1,awg2);
        end

    end

end
