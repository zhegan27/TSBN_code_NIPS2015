
load( 'mocap_1.mat' )

W1=param{1}; % J*J*nt
W2=param{2}; % M*J
W2prime=param{3}; % M*J
W3=param{4}; % J*M*nt
W4=param{5}; % M*M*nt
W4prime=param{6}; % M*M*nt

U1=param{7}; % J*J*nt
U2=param{8}; % J*M
U3=param{9}; % J*M*nt

b1=param{10}; % J*1
b2=param{11}; % M*1
b3=param{12}; % M*1
c=param{13}; % J*1

% v = TrainData{38}'; % waliking
 v = TrainData{4}'; % running

[~,J,nt] = size(W1); [M,T] = size(v);

%% feed-forward step
h = zeros(J,T);
vrec = zeros(M,T);

% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+c)>rand(J,1));
vrec(:,1) = W2*h(:,1)+b2;

for t = 2:T
    cc = zeros(J,1); bb2 = zeros(M,1);
    for delay = 1:min(t-1,nt)
        cc = cc + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*v(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U2*v(:,t)+cc+c)>rand(J,1));
    vrec(:,t) = W2*h(:,t)+bb2+b2;
end;


%% generate step
T = 400;
hgen = zeros(J,T); vgen = zeros(M,T);
% sampling
hgen(:,1:10) = h(:,1:10);
vgen(:,1:10) = v(:,1:10);

for t = 11:T
    bb1 = zeros(J,1); bb2 = zeros(M,1);
    for delay = 1:min(t-1,nt)
        bb1 = bb1 + W1(:,:,delay)*hgen(:,t-delay)+W3(:,:,delay)*vgen(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*vgen(:,t-delay);
    end;
    hgen(:,t) = double(sigmoid(bb1+b1)>rand(J,1));
    vgen(:,t) = W2*hgen(:,t)+bb2+b2;
end;

