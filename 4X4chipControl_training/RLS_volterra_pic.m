function [k,W1,W2,W3] = RLS_volterra_pic(taps_linear,taps_quadratic,taps_cubic,numofTs,Rxdata,Txdata)

Rxdata = reshape(Rxdata,1,[]).';   % received data
Rxdata = Rxdata-mean(Rxdata);
Rxdata = Rxdata/sqrt(mean(abs(Rxdata).^2));

Txdata1 = reshape(Txdata,1,[]).';    % original data
% Txdata=qammod(Txdata,2^Bitpersymbol);
Txdata1 = Txdata1-mean(Txdata1);
Txdata1 = Txdata1/sqrt(mean(abs(Txdata1).^2));

x = Rxdata(256:numofTs);   % training symbol
d = Txdata1(256:numofTs);
% d=upsample(d,2);

[Mx,Nx] = size(x);

Sx = [Mx,Nx];
ntr = max(Sx);              %  temporary number of iterations
compensation = max(max(taps_linear,taps_quadratic),taps_cubic);
y = zeros(length(Sx),1);              %  initialize output signal vector
E = zeros(length(Sx),1);              %  initialize error signal vector

% 
% u_Linear = 0.0033;
% u_quadratic = 0.0005;
% u_cubic = 0.0005;
delta = 0.00001;
num_quadratic = taps_quadratic*(taps_quadratic-1)/2+taps_quadratic;
num_cubic = (taps_cubic+taps_cubic*(taps_cubic-1)/2)*taps_cubic;
% W1 = zeros(taps_linear,1);
W1 = zeros(taps_linear,1);
W1(1:(taps_linear-1)/2)=0;   % taps of LE
W1((taps_linear+3)/2:taps_linear)=0;
W1((taps_linear+1)/2)=1;
W2 = delta*ones(num_quadratic,1);   % taps of quadratic term
W3 = delta*ones(num_cubic,1);   % taps of cubic term
W = [W1;W2;W3];
P = eye(taps_linear+num_quadratic+num_cubic);
forget_factor = 0.9999;
%%%%%%
%  Main loop
nn=1;
for i=1
    n=compensation;
   
    while n<=ntr       
        X_linear = x(n-compensation+(compensation+taps_linear)/2:-1:n-compensation+(compensation+1-taps_linear+1)/2);
        X_quadratic = x(n-compensation+(compensation+taps_quadratic)/2:-1:n-compensation+(compensation+1-taps_quadratic+1)/2);
        X_cubic = x(n-compensation+(compensation+taps_cubic)/2:-1:n-compensation+(compensation+1-taps_cubic+1)/2);
        X_q = triu(X_quadratic*X_quadratic');
        X_q(X_q==0) = [];
        X_q = reshape(X_q,1,[]).';
        X_c2 = triu(X_cubic*X_cubic');
        X_c2(X_c2==0) = [];
        X_c2 = reshape(X_c2,1,[]).';
        X_c = X_c2*X_cubic';
        X_c = reshape(X_c,1,[]).';
        X = [X_linear;X_q;X_c];
%         y(nn) = W1'*X_linear+W2'*X_q+W3'*X_c;
        temp_x=P*X;
        G=temp_x/(forget_factor+X'*temp_x);
        temp_output=W'*X;
        e = d(n-compensation+ceil((compensation+1)/2))-temp_output;
        W=W+G.*conj(e);
        P=1/forget_factor*(P-G*(X'*P));
        E(nn)=e;     % error        
        n=n+1;
        nn=nn+1;
    end
end
% figure();
% plot(E.*conj(E));
% y=y/sqrt(mean(abs(y).^2));
% figure;bar(W);  % 画出抽头分布
% figure;bar3(V);
% save W.txt W -ascii;
n=compensation;
mm=1;
while n<=length(Txdata1)
    R_linear = Rxdata(n-compensation+(compensation+taps_linear)/2:-1:n-compensation+(compensation+1-taps_linear+1)/2);
    R_quadratic = Rxdata(n-compensation+(compensation+taps_quadratic)/2:-1:n-compensation+(compensation+1-taps_quadratic+1)/2);
    R_cubic = Rxdata(n-compensation+(compensation+taps_cubic)/2:-1:n-compensation+(compensation+1-taps_cubic+1)/2);
    R_q = triu(R_quadratic*R_quadratic');
    R_q(R_q==0) = [];
    R_q = reshape(R_q,1,[]).';
    R_c2 = triu(R_cubic*R_cubic');
    R_c2(R_c2==0) = [];
    R_c2 = reshape(R_c2,1,[]).';
    R_c = R_c2*R_cubic';
    R_c = reshape(R_c,1,[]).';
    R = [R_linear;R_q;R_c];
    k(mm) = W'*R;
    n=n+1;
    mm=mm+1;
end

W1 = W(1:taps_linear);
W2 = W(taps_linear+1:taps_linear+num_quadratic);
W3 = W(num_quadratic+taps_linear+1:taps_linear+num_quadratic+num_cubic);
k=[Rxdata(1:(compensation-1)/2).' k Rxdata(end-(compensation-1)/2+1:end).'];
% k=[Rxdata(1:(compensation-1)).' k]; %补偿抽头损失

k=k*std(Txdata)+mean(Txdata);   % output after RLS+volterra

% figure();
% plot(abs(E),'b-');title('error');
