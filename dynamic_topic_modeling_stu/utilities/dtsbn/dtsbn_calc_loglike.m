
function [lb, pr,rc ]= dtsbn_calc_loglike(vhdout,vtrain,parameters)

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
U1=parameters{3}; % J*J
U2=parameters{4}; % J*M
b1=parameters{5}; % J*1
b2=parameters{6}; % M*1
c=parameters{7}; % J*1
binit=parameters{8}; % J*1
cinit=parameters{9}; % J*1

[~,J] = size(W1); [M,T] = size(vhdout);

%% feed-forward step of heldout
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*vhdout(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*vhdout(:,t)+c)>rand(J,1));
end;

%% calculate lower bound of heldout
term1 = [binit,bsxfun(@plus,W1*h(:,1:T-1),b1)]; % J*T
term2 = bsxfun(@plus,W2*h,b2); % M*T
term3 = [U2*vhdout(:,1)+cinit, ...
    bsxfun(@plus,U1*h(:,1:T-1)+U2*vhdout(:,2:T),c)]; % J*T

logprior = sum(term1.*h-log(1+exp(term1))); % 1*T
loglike = sum(term2.*vhdout-bsxfun(@times,log(logsumexp(term2)), vhdout )); % 1*T
logpost = sum(term3.*h-log(1+exp(term3))); % 1*T
ll = logprior + loglike - logpost; % 1*T
lb = mean(ll);


%% prediction of heldout
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*vtrain(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*vtrain(:,t)+c)>rand(J,1));
end;

term2 = bsxfun(@plus,W2*h,b2); % M*T 
probpred = decsoftmax(term2);
[~,cpred] = sort(probpred, 1,'descend' );
[~,ctrue] = sort(vhdout, 1, 'descend' );

range = 50;  
for t = 1:T
    g1 = zeros(M,1); g2 = zeros(M,1);
    g1(ctrue(1:range, t),1) = 1;
    g2(cpred(1:range, t),1) = 1;
    con = confusionmat(g1,g2);
    TNR(t) = con(2,2)/(con(2,2)+con(1,2));
    T = find( vhdout(:,t) ~= 0  );
    C = intersect(T, cpred(1:range, t) );
    % C = setdiff(cpred(1:range, t), intersect(T, cpred(1:range, t) ));
    TPR(t) = length(C)/length(T);
end
pr = mean(TNR); rc =  mean(TPR);
end        

