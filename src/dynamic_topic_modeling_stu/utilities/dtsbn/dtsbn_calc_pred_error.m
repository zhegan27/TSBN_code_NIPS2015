
function [rec_err,pp] = dtsbn_calc_pred_error(vtest,vtrain,parameters)

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
U1=parameters{3}; % J*J
U2=parameters{4}; % J*M
b1=parameters{5}; % J*1
b2=parameters{6}; % M*1
c=parameters{7}; % J*1
binit=parameters{8}; % J*1
cinit=parameters{9}; % J*1

[~,J] = size(W1); [~,T] = size(vtrain);

rec_err = 0; pred_err = 0; Nsample = 1;

for iter = 1:Nsample
%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*vtrain(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*vtrain(:,t)+c)>rand(J,1));
end;

%% normailization reconstruction
term2 = bsxfun(@plus,W2*h,b2); % M*T 
rec_prob = decsoftmax(term2);
rec_err = rec_err + mean(sum(( bsxfun(@times, vtrain(:,2:T), 1./sum(vtrain(:,2:T),1) ) -rec_prob(:,2:T)).^2))/Nsample;

%% prediction step
term1 = bsxfun(@plus,W1*h(:,T),b1); % J
htest = double(sigmoid(term1)>rand(J,1));
term2 = bsxfun(@plus,W2*htest,b2); % M*T 
probpred = decsoftmax(term2);
[~,cpred] = sort(probpred, 'descend' );

[~,ctrue] = sort(vtest, 'descend' );
range = 50; c = intersect(cpred(1:range),ctrue(1:range)) ;
pp = length(c)/range;

end        

