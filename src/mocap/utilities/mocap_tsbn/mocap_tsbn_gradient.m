
function [gradient,lb,meanll,varll] = mocap_tsbn_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

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

A1 = parameters{14}; % 1*L
A2 = parameters{15}; % L*M

[~,J,nt] = size(W1); [M,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+c)>rand(J,1));

for t = 2:T
    cc = zeros(J,1);
    for delay = 1:min(t-1,nt)
        cc = cc + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U2*v(:,t)+cc+c)>rand(J,1));
end;

%% calculate lower bound
term1 = zeros(J,T);
mu = zeros(M,T); logsigma = zeros(M,T);
term3 = zeros(J,T);

% sampling
term1(:,1) = b1;
mu(:,1) = W2*h(:,1)+b2;
logsigma(:,1) = W2prime*h(:,1)+b3;
term3(:,1) = U2*v(:,1)+c;

for t = 2:T
    bb1 = zeros(J,1); bb2 = zeros(M,1);
    bb3 = zeros(M,1); cc = zeros(J,1);
    for delay = 1:min(t-1,nt)
        bb1 = bb1 + W1(:,:,delay)*h(:,t-delay)+W3(:,:,delay)*v(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*v(:,t-delay);
        bb3 = bb3 + W4prime(:,:,delay)*v(:,t-delay);
        cc = cc + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
    end;
    term1(:,t) = bb1+b1;
    mu(:,t) = W2*h(:,t)+bb2+b2;
    logsigma(:,t) = W2prime*h(:,t)+bb3+b3;
    term3(:,t) = U2*v(:,t)+cc+c;
end;

logprior = sum(term1.*h-log(1+exp(term1))); % 1*T
loglike = -sum(logsigma + (v-mu).^2./(2*exp(2*logsigma))+1/2*log(2*pi)); % 1*T
logpost = sum(term3.*h-log(1+exp(term3))); % 1*T

ll = logprior + loglike - logpost; % 1*T
lb = mean(ll);

ll = ll - A1*tanh(A2*v);

if prevMean == 0 && prevVar == 0
    meanll = mean(ll);
    varll = var(ll);
else
    meanll = alpha*prevMean + (1-alpha)*mean(ll);
    varll = alpha*prevVar + (1-alpha)*var(ll);
end;

ll = (ll-meanll)./max(1,sqrt(varll));

%% gradient information
mat1 = h - sigmoid(term1); % J*T
grads.W1 = zeros(J,J,nt); grads.W3 = zeros(J,M,nt);
for delay = 1:nt
    grads.W1(:,:,delay) = mat1(:,delay+1:T)*h(:,1:T-delay)'/(T-delay); % J*J
    grads.W3(:,:,delay) = mat1(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % J*M
end;
grads.b1 = sum(mat1,2)/T; % J*1


mat2 = (v-mu)./(exp(2*logsigma)); % M*T
mat2p = (v-mu).^2./(exp(2*logsigma))-1; % M*T
grads.W2 = mat2*h'/T ; % M*J
grads.W2prime = mat2p*h'/T ; % M*J
grads.W4 = zeros(M,M,nt);
grads.W4prime = zeros(M,M,nt);
for delay = 1:nt
    grads.W4(:,:,delay) = mat2(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
    grads.W4prime(:,:,delay) = mat2p(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
end;
grads.b2 = sum(mat2,2)/T; % M*1
grads.b3 = sum(mat2p,2)/T; % M*1

mat3 = bsxfun(@times,h-sigmoid(term3),ll); % J*T
grads.U2 = mat3*v'/T; % J*M
grads.U1 = zeros(J,J,nt); grads.U3 = zeros(J,M,nt);
for delay = 1:nt
    grads.U1(:,:,delay) = mat3(:,delay+1:T)*h(:,1:T-delay)'/(T-delay); % J*J
    grads.U3(:,:,delay) = mat3(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % J*M
end;
grads.c = sum(mat3,2)/T; % J*1

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'/T; % 1*L
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v'/T;



%% collection
gradient{1} = grads.W1;
gradient{2} = grads.W2;
gradient{3} = grads.W2prime;
gradient{4} = grads.W3;
gradient{5} = grads.W4;
gradient{6} = grads.W4prime;

gradient{7} = grads.U1;
gradient{8} = grads.U2;
gradient{9} = grads.U3;

gradient{10} = grads.b1;
gradient{11} = grads.b2;
gradient{12} = grads.b3;
gradient{13} = grads.c;

gradient{14} = grads.A1;
gradient{15} = grads.A2;

end
