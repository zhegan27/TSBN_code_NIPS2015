
function [rec_err,pred_err] = tsbn_calc_pred_error(v,parameters)

W1 = parameters{1}; % J*J
W2 = parameters{2}; % M*J
W3 = parameters{3}; % J*M
W4 = parameters{4}; % M*M

U1 = parameters{5}; % J*J
U2 = parameters{6}; % J*M
U3 = parameters{7}; % J*M

b  = parameters{8}; % J*1
c  = parameters{9}; % M*1
d  = parameters{10}; % J*1

binit = parameters{11}; % J*1
cinit = parameters{12}; % M*1
dinit = parameters{13}; % J*1

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+dinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+U3*v(:,t-1)+d)>rand(J,1));
end;

%% reconstruction
vrec = [sigmoid(W2*h(:,1)+cinit),sigmoid(bsxfun(@plus,W2*h(:,2:T)+W4*v(:,1:T-1),c))];
% vrec = double(vrec>rand(size(vrec)));
rec_err = mean(sum((v(:,2:T)-vrec(:,2:T)).^2));

%% prediction step

hpred = [sigmoid(binit),sigmoid(bsxfun(@plus,W1*h(:,1:T-1)+W3*v(:,2:T),b))];
hpred = double(hpred>rand(size(hpred)));

vpred = [sigmoid(W2*hpred(:,1)+cinit),...
    sigmoid(bsxfun(@plus,W2*hpred(:,2:T)+W4*v(:,1:T-1),c))];
% vpred = double(vpred>rand(size(vpred)));

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));

end        

