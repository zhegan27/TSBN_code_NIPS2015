
function [rec_err,pred_err] = dtsbn_deter_calc_pred_error(v,parameters)

W1 = parameters{1}; % J*J*nt
W2 = parameters{2}; % K*J
W4 = parameters{3}; % K*K*nt
W5 = parameters{4}; % M*K
W6 = parameters{5}; % K*M*nt
W7 = parameters{6}; % M*M*nt

U1 = parameters{7}; % J*J*nt
U2 = parameters{8}; % J*K
U4 = parameters{9}; % K*K*nt
U5 = parameters{10}; % K*M
U6 = parameters{11}; % K*M*nt

b1  = parameters{12}; % J*1
b2  = parameters{13}; % K*1
b3  = parameters{14}; % M*1
c1  = parameters{15}; % J*1
c2  = parameters{16}; % K*1

[~,J,nt] = size(W1); [K,~] = size(W2); [M,T] = size(v);

%% feed-forward step
hr = zeros(K,T); z = zeros(J,T); hg = zeros(K,T); vrec = zeros(M,T);
% sampling
hr(:,1) = max(U5*v(:,1)+c2,0);
z(:,1) = double(sigmoid(U2*hr(:,1)+c1)>rand(J,1));
hg(:,1) = max(W2*z(:,1)+b2,0);
vrec(:,1) = sigmoid(W5*hg(:,1)+b3);

for t = 2:T
    cc2 = zeros(K,1);  cc1 = zeros(J,1);  bb2 = zeros(K,1); bb3 = zeros(M,1);
    for delay = 1:min(t-1,nt) 
        cc2 = cc2 + U4(:,:,delay)*hr(:,t-delay)+U6(:,:,delay)*v(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*hg(:,t-delay)+W6(:,:,delay)*v(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*v(:,t-delay);
    end;
    hr(:,t) = max(U5*v(:,t)+cc2+c2,0);
    z(:,t) = double(sigmoid(U2*hr(:,t)+cc1+c1)>rand(J,1));
    hg(:,t) = max(W2*z(:,t)+bb2+b2,0);
    vrec(:,t) = sigmoid(W5*hg(:,t)+bb3+b3);
end;

%% reconstruction

rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step
zpred = zeros(J,T); hgpred = zeros(K,T); vpred = zeros(M,T);
% sampling
zpred(:,1) = double(sigmoid(b1)>rand(J,1));
hgpred(:,1) = max(W2*zpred(:,1)+b2,0);
vpred(:,1) = sigmoid(W5*hgpred(:,1)+b3);

for t = 2:T
    bb1 = zeros(J,1);  bb2 = zeros(K,1); bb3 = zeros(M,1);
    for delay = 1:min(t-1,nt) 
        bb1 = bb1 + W1(:,:,delay)*z(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*hg(:,t-delay)+W6(:,:,delay)*v(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*v(:,t-delay);
    end;
    zpred(:,t) = double(sigmoid(bb1+b1)>rand(J,1));
    hgpred(:,t) = max(W2*zpred(:,t)+bb2+b2,0);
    vpred(:,t) = sigmoid(W5*hgpred(:,t)+bb3+b3);
end;

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));

end        

