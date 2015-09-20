
function [gradient,lb,meanll,varll] = tsbn_delay_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

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

A1 = parameters{11}; % 1*L
A2 = parameters{12}; % L*M

[~,J,nt] = size(W1); [M,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+d)>rand(J,1));

for t = 2:T
    bb = zeros(J,1);
    for delay = 1:min(t-1,nt)
        bb = bb + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U2*v(:,t)+bb+d)>rand(J,1));
end;

%% calculate lower bound 
term1 = zeros(J,T); term2 = zeros(M,T); term3 = zeros(J,T);
% sampling
term1(:,1) = b;
term2(:,1) = W2*h(:,1)+c;
term3(:,1) = U2*v(:,1)+d;

for t = 2:T
    bb = zeros(J,1);
    cc = zeros(M,1); dd = zeros(J,1);
    for delay = 1:min(t-1,nt)
        bb = bb + W1(:,:,delay)*h(:,t-delay)+W3(:,:,delay)*v(:,t-delay);
        cc = cc + W4(:,:,delay)*v(:,t-delay);
        dd = dd + U1(:,:,delay)*h(:,t-delay)+U3(:,:,delay)*v(:,t-delay);
    end;
    term1(:,t) = bb+b;
    term2(:,t) = W2*h(:,t)+cc+c;
    term3(:,t) = U2*v(:,t)+dd+d;
end;

logprior = sum(term1.*h-log(1+exp(term1))); % 1*T
loglike = sum(term2.*v-log(1+exp(term2))); % 1*T
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
grads.b = sum(mat1,2)/T; % J*1


mat2 = v-sigmoid(term2); % M*T
grads.W2 = mat2*h'/T ; % M*J
grads.W4 = zeros(M,M,nt);
for delay = 1:nt
    grads.W4(:,:,delay) = mat2(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
end;
grads.c = sum(mat2,2)/T; % M*1

mat3 = bsxfun(@times,h-sigmoid(term3),ll); % J*T
grads.U2 = mat3*v'/T; % J*M
grads.U1 = zeros(J,J,nt); grads.U3 = zeros(J,M,nt);
for delay = 1:nt
    grads.U1(:,:,delay) = mat3(:,delay+1:T)*h(:,1:T-delay)'/(T-delay); % J*J
    grads.U3(:,:,delay) = mat3(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % J*M
end;
grads.d = sum(mat3,2)/T; % J*1 

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'/T; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v'/T;

%% collection
gradient{1} = grads.W1;
gradient{2} = grads.W2;
gradient{3} = grads.W3;
gradient{4} = grads.W4;

gradient{5} = grads.U1;
gradient{6} = grads.U2;
gradient{7} = grads.U3;

gradient{8} = grads.b;    
gradient{9} = grads.c; 
gradient{10} = grads.d; 

gradient{11} = grads.A1; 
gradient{12} = grads.A2; 

end        
