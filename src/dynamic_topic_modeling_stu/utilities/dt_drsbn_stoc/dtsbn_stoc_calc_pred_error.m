
function [rec_err,pp] = dtsbn_stoc_calc_pred_error(vtest,vtrain,parameters)

W1 = parameters{1}; % J*J*nt
W2 = parameters{2}; % K*J
W3 = parameters{3}; % J*K*nt
W4 = parameters{4}; % K*K*nt
W5 = parameters{5}; % M*K
W6 = parameters{6}; % K*M*nt
W7 = parameters{7}; % M*M*nt

U1 = parameters{8}; % J*J*nt
U2 = parameters{9}; % J*K
U3 = parameters{10}; % J*K*nt
U4 = parameters{11}; % K*K*nt
U5 = parameters{12}; % K*M
U6 = parameters{13}; % K*M*nt

b1  = parameters{14}; % J*1
b2  = parameters{15}; % K*1
b3  = parameters{16}; % M*1
c1  = parameters{17}; % J*1
c2  = parameters{18}; % K*1

[~,J,nt] = size(W1); [K,~] = size(W2); [M,T] = size(vtrain);

%% feed-forward step
h = zeros(K,T); z = zeros(J,T); vrec = zeros(M,T);
% sampling
h(:,1) = double(sigmoid(U5*vtrain(:,1)+c2)>rand(K,1));
z(:,1) = double(sigmoid(U2*h(:,1)+c1)>rand(J,1));
vrec(:,1) = sigmoid(W5*h(:,1)+b3);

for t = 2:T
    cc2 = zeros(K,1);  cc1 = zeros(J,1); bb3 = zeros(M,1);
    for delay = 1:min(t-1,nt) 
        cc2 = cc2 + U4(:,:,delay)*h(:,t-delay)+U6(:,:,delay)*vtrain(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay)+U3(:,:,delay)*h(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*vtrain(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U5*vtrain(:,t)+cc2+c2)>rand(K,1));
    z(:,t) = double(sigmoid(U2*h(:,t)+cc1+c1)>rand(J,1));
    vrec(:,t) = sigmoid(W5*h(:,t)+bb3+b3);
end;

%% reconstruction
%% normailization reconstruction
term2 = bsxfun(@plus,W5*h,b3); % M*T 
rec_prob = decsoftmax(term2);
rec_err =  mean(sum(( bsxfun(@times, vtrain(:,2:T), 1./sum(vtrain(:,2:T),1) ) -rec_prob(:,2:T)).^2));


%% prediction step
zpred = zeros(J,T); hpred = zeros(K,T); vpred = zeros(M,T);
% sampling
zpred(:,1) = double(sigmoid(b1)>rand(J,1));
hpred(:,1) = double(sigmoid(W2*zpred(:,1)+b2)>rand(K,1));
vpred(:,1) = sigmoid(W5*hpred(:,1)+b3);

for t = 2:T
    bb1 = zeros(J,1);  bb2 = zeros(K,1); bb3 = zeros(M,1);
    for delay = 1:min(t-1,nt) 
        bb1 = bb1 + W1(:,:,delay)*z(:,t-delay)+W3(:,:,delay)*h(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*h(:,t-delay)+W6(:,:,delay)*vtrain(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*vtrain(:,t-delay);
    end;
    zpred(:,t) = double(sigmoid(bb1+b1)>rand(J,1));
    hpred(:,t) = double(sigmoid(W2*zpred(:,t)+bb2+b2)>rand(K,1));
    vpred(:,t) = sigmoid(W5*hpred(:,t)+bb3+b3);
end;

pred_err = mean(sum((vtrain(:,2:T)-vpred(:,2:T)).^2));


%% prediction step
bb1 = zeros(J,1);  bb2 = zeros(K,1); bb3 = zeros(M,1);
for delay = 1:min(T-1,nt) 
    bb1 = bb1 + W1(:,:,delay)*z(:,T+1-delay)+W3(:,:,delay)*h(:,T+1-delay);
    bb2 = bb2 + W4(:,:,delay)*h(:,T+1-delay)+W6(:,:,delay)*vtrain(:,T+1-delay);
    bb3 = bb3 + W7(:,:,delay)*vtrain(:,T+1-delay);
end;
zpred = double(sigmoid(bb1+b1)>rand(J,1));
hpred = double(sigmoid(W2*zpred+bb2+b2)>rand(K,1));
probpred = decsoftmax(W5*hpred+bb3+b3);
[~,cpred] = sort(probpred, 'descend' );

[~,ctrue] = sort(vtest, 'descend' );
range = 50; c = intersect(cpred(1:range),ctrue(1:range)) ;
pp = length(c)/range;

end        

