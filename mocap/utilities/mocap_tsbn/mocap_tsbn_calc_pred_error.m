
function [rec_err,pred_err] = mocap_tsbn_calc_pred_error(v,parameters)

W1=parameters{1}; % J*J*nt
W2=parameters{2}; % M*J
W2prime=parameters{3}; % M*J
W3=parameters{4}; % J*M*nt
W4=parameters{5}; % M*M*nt
W4prime=parameters{6}; % M*M*nt

U1=parameters{7}; % J*J*nt
U2=parameters{8}; % J*M
U3=parameters{9}; % J*M*nt

b1=parameters{10}; % J*1
b2=parameters{11}; % M*1
b3=parameters{12}; % M*1
c=parameters{13}; % J*1

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

%% reconstruction
rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step
hpred = zeros(J,T); vpred = zeros(M,T);
% sampling
hpred(:,1) = double(sigmoid(b1)>rand(J,1));
vpred(:,1) = sigmoid(W2*hpred(:,1)+b2);

for t = 2:T
    bb1 = zeros(J,1); bb2 = zeros(M,1);
    for delay = 1:min(t-1,nt)
        bb1 = bb1 + W1(:,:,delay)*h(:,t-delay)+W3(:,:,delay)*v(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*v(:,t-delay);
    end;
    hpred(:,t) = double(sigmoid(bb1+b1)>rand(J,1));
    vpred(:,t) = W2*hpred(:,t)+bb2+b2;
end;

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));


end
