
function [rec_err,pred_err] = tsbn_delay_calc_pred_error(v,parameters)

W1 = parameters{1}; % J*J*nt
W2 = parameters{2}; % M*J
W3 = parameters{3}; % J*M*nt
W4 = parameters{4}; % M*M*nt

U1 = parameters{5}; % J*J*nt
U2 = parameters{6}; % J*M
U3 = parameters{7}; % J*M*nt

b  = parameters{8}; % J*1
c  = parameters{9}; % M*1
d  = parameters{10}; % J*1

[~,J,nt] = size(W1); [M,T] = size(v);

%% feed-forward step
h = zeros(J,T); vrec = zeros(M,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+d)>rand(J,1));
vrec(:,1) = sigmoid(W2*h(:,1)+c);

for t = 2:T
    bb = zeros(J,1);
    cc = zeros(M,1);
    for delay = 1:min(t-1,nt)
        bb = bb + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
        cc = cc + W4(:,:,delay)*v(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U2*v(:,t)+bb+d)>rand(J,1));
    vrec(:,t) = sigmoid(W2*h(:,t)+cc+c);
end;

%% reconstruction

rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step
hpred = zeros(J,T); vpred = zeros(M,T);
% sampling
hpred(:,1) = double(sigmoid(b)>rand(J,1));
vpred(:,1) = sigmoid(W2*hpred(:,1)+c);

for t = 2:T
    bb = zeros(J,1);
    cc = zeros(M,1);
    for delay = 1:min(t-1,nt)
        bb = bb + W1(:,:,delay)*h(:,t-delay)+W3(:,:,delay)*v(:,t-delay);
        cc = cc + W4(:,:,delay)*v(:,t-delay);
    end;
    hpred(:,t) = double(sigmoid(bb+b)>rand(J,1));
    vpred(:,t) = sigmoid(W2*hpred(:,t)+cc+c);
end;

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));

end        

