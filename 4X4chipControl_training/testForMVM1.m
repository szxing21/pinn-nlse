
CTool = ConvTool;

rng(12)
Mat1 = rand([4,4]);
rng(123)
X1 = rand([4,50]);
idealOut = Mat1 * X1;
B = [0;0;0;0];

%%
[dac, tekAWG, agAWG, pd1, pd2, laser] = initInstr;

%%
[out1, outValCol] = CTool.multiSampMVM_ver2(X1,Mat1,B,tekAWG,agAWG,laser,pd1,pd2,dac);
