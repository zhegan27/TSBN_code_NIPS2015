
function [gradient,lb,meanll,varll] = mocap_dtsbn_deter_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

W1 = parameters{1}; % J*J*nt
W2 = parameters{2}; % K*J
W4 = parameters{3}; % K*K*nt
W5 = parameters{4}; % M*K
W5prime = parameters{5}; % M*K
W6 = parameters{6}; % K*M*nt
W7 = parameters{7}; % M*M*nt
W7prime = parameters{8}; % M*M*nt

U1 = parameters{9}; % J*J*nt
U2 = parameters{10}; % J*K
U4 = parameters{11}; % K*K*nt
U5 = parameters{12}; % K*M
U6 = parameters{13}; % K*M*nt

b1  = parameters{14}; % J*1
b2  = parameters{15}; % K*1
b3  = parameters{16}; % M*1
b3prime  = parameters{17}; % M*1
c1  = parameters{18}; % J*1
c2  = parameters{19}; % K*1

A1 = parameters{20}; % 1*L
A2 = parameters{21}; % L*M

[~,J,nt] = size(W1); [K,~] = size(W2); [M,T] = size(v);

%% feed-forward step
hr = zeros(K,T); z = zeros(J,T); hg = zeros(K,T);
hr_diff = zeros(K,T); hg_diff = zeros(K,T);
% sampling
hr(:,1) = max(U5*v(:,1)+c2,0);
hr_diff(:,1) = double(U5*v(:,1)+c2>0);
z(:,1) = double(sigmoid(U2*hr(:,1)+c1)>rand(J,1));
hg(:,1) = max(W2*z(:,1)+b2,0);
hg_diff(:,1) = double(W2*z(:,1)+b2>0);

for t = 2:T
    cc2 = zeros(K,1);  cc1 = zeros(J,1);  bb2 = zeros(K,1);
    for delay = 1:min(t-1,nt) 
        cc2 = cc2 + U4(:,:,delay)*hr(:,t-delay)+U6(:,:,delay)*v(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*hg(:,t-delay)+W6(:,:,delay)*v(:,t-delay);
    end;
    hr(:,t) = max(U5*v(:,t)+cc2+c2,0);
    hr_diff(:,t) = double(U5*v(:,t)+cc2+c2>0);
    z(:,t) = double(sigmoid(U2*hr(:,t)+cc1+c1)>rand(J,1));
    hg(:,t) = max(W2*z(:,t)+bb2+b2,0);
    hg_diff(:,t) = double(W2*z(:,t)+bb2+b2>0);
end;

%% calculate lower bound 
term1 = zeros(J,T); 
mu = zeros(M,T); logsigma = zeros(M,T);
term3 = zeros(J,T);
% sampling
term1(:,1) = b1;
mu(:,1) = W5*hg(:,1)+b3;
logsigma(:,1) = W5prime*hg(:,1)+b3prime;
term3(:,1) = U2*hr(:,1)+c1;

for t = 2:T
    bb1 = zeros(J,1); bb3prime = zeros(M,1);
    bb3 = zeros(M,1); cc1 = zeros(J,1);
    for delay = 1:min(t-1,nt)
        bb1 = bb1 + W1(:,:,delay)*z(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*v(:,t-delay);
        bb3prime = bb3prime + W7prime(:,:,delay)*v(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay);
    end;
    term1(:,t) = bb1+b1;
    mu(:,t) = W5*hg(:,t)+bb3+b3;
    logsigma(:,t) = W5prime*hg(:,t)+bb3prime+b3prime;
    term3(:,t) = U2*hr(:,t)+cc1+c1;
end;

logprior = sum(term1.*z-log(1+exp(term1))); % 1*T
loglike = -sum(logsigma + (v-mu).^2./(2*exp(2*logsigma))+1/2*log(2*pi)); % 1*T
logpost = sum(term3.*z-log(1+exp(term3))); % 1*T
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
% model parameters
mat1 = z - sigmoid(term1); % J*T
grads.W1 = zeros(J,J,nt);
for delay = 1:nt
    grads.W1(:,:,delay) = mat1(:,delay+1:T)*z(:,1:T-delay)'/(T-delay); % J*J
end;
grads.b1 = sum(mat1,2)/T; % J*1

mat2 = (v-mu)./(exp(2*logsigma)); % M*T
mat2p = (v-mu).^2./(exp(2*logsigma))-1; % M*T
grads.W5 = mat2*hg'/T ; % M*K
grads.W5prime = mat2p*hg'/T ; % M*J
grads.W7 = zeros(M,M,nt);
grads.W7prime = zeros(M,M,nt);
for delay = 1:nt
    grads.W7(:,:,delay) = mat2(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
    grads.W7prime(:,:,delay) = mat2p(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
end;
grads.b3 = sum(mat2,2)/T; % M*1
grads.b3prime = sum(mat2p,2)/T; % M*1

grad_hg = zeros(K,T);
grad_hg(:,T) = W5'*mat2(:,T)+W5prime'*mat2p(:,T); % K*1
for t = T-1:1
    grad_hg(:,t) = W4(:,:,1)*(grad_hg(:,t+1).*hg_diff(:,t+1)) +...
        W5'*mat2(:,t) + W5prime'*mat2p(:,t);
end;
grad_hg = grad_hg.* hg_diff; % K*T

grads.W2 = grad_hg*z'/T; % K*J
grads.W4 = zeros(K,K,nt); grads.W6 = zeros(K,M,nt);
for delay = 1:nt
    grads.W4(:,:,delay) = grad_hg(:,delay+1:T)*hg(:,1:T-delay)'/(T-delay); % K*K
    grads.W6(:,:,delay) = grad_hg(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % K*M
end;
grads.b2 = sum(grad_hg,2)/T; % M*1

% recognition parameters
mat3 = bsxfun(@times,z-sigmoid(term3),ll); % J*T
grads.U2 = mat3*hr'/T; % J*K
grads.U1 = zeros(J,J,nt); 
for delay = 1:nt
    grads.U1(:,:,delay) = mat3(:,delay+1:T)*z(:,1:T-delay)'/(T-delay); % J*J
end;
grads.c1 = sum(mat3,2)/T; % J*1 

grad_hr = zeros(K,T);
grad_hr(:,T) = U2'*mat3(:,T); % K*1
for t = T-1:1
    grad_hr(:,t) = U4(:,:,1)*(grad_hr(:,t+1).*hr_diff(:,t+1)) + U2'*mat3(:,t);
end;
grad_hr = grad_hr.* hr_diff; % K*T

grads.U5 = grad_hr*v'/T; % K*M
grads.U4 = zeros(K,K,nt); grads.U6 = zeros(K,M,nt);
for delay = 1:nt
    grads.U4(:,:,delay) = grad_hr(:,delay+1:T)*hr(:,1:T-delay)'/(T-delay); % K*K
    grads.U6(:,:,delay) = grad_hr(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % K*M
end;
grads.c2 = sum(grad_hr,2)/T; % K*1

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'/T; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v'/T;

%% collection
gradient{1} = grads.W1;
gradient{2} = grads.W2;
gradient{3} = grads.W4;
gradient{4} = grads.W5;
gradient{5} = grads.W5prime;
gradient{6} = grads.W6;
gradient{7} = grads.W7;
gradient{8} = grads.W7prime;

gradient{9} = grads.U1;
gradient{10} = grads.U2;
gradient{11} = grads.U4;
gradient{12} = grads.U5;
gradient{13} = grads.U6;

gradient{14} = grads.b1;    
gradient{15} = grads.b2; 
gradient{16} = grads.b3; 
gradient{17} = grads.b3prime; 
gradient{18} = grads.c1; 
gradient{19} = grads.c2; 

gradient{20} = grads.A1; 
gradient{21} = grads.A2; 

end        
