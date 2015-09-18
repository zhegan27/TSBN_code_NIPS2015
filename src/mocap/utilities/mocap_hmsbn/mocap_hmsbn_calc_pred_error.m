
function [rec_err,pred_err] = mocap_hmsbn_calc_pred_error(v,parameters)

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
W3=parameters{3}; % M*J

U1=parameters{4}; % J*J
U2=parameters{5}; % J*M

b1=parameters{6}; % J*1
b2=parameters{7}; % M*1
b3=parameters{8}; % M*1
c=parameters{9}; % J*1

binit=parameters{10}; % J*1
cinit=parameters{11}; % J*1

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+c)>rand(J,1));
end;

%% reconstruction

vrec=bsxfun(@plus,b2,W2*h);

rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step
hpred = zeros(J,T);
hpred(:,1) = double(sigmoid(binit)>rand(J,1));
for t = 2:T
    hpred(:,t) = double(sigmoid(W1*h(:,t-1)+b1)>rand(J,1));
end;

vpred = bsxfun(@plus,W2*hpred,b2);

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));


end        

