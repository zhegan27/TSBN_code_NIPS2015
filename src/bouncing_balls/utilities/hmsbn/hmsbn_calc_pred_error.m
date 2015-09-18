
function [rec_err,pred_err] = hmsbn_calc_pred_error(v,parameters)

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
U1=parameters{3}; % J*J
U2=parameters{4}; % J*M
b=parameters{5}; % J*1
c=parameters{6}; % M*1
d=parameters{7}; % J*1
binit=parameters{8}; % J*1
dinit=parameters{9}; % J*1

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+dinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+d)>rand(J,1));
end;

%% reconstruction

vrec=sigmoid(bsxfun(@plus,c,W2*h));
rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step

hpred = [sigmoid(binit),sigmoid(bsxfun(@plus,W1*h(:,1:T-1),b))];
hpred = double(hpred>rand(size(hpred)));

vpred = sigmoid(bsxfun(@plus,W2*hpred,c));
pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));


end        

