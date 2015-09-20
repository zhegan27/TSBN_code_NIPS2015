
function [gradient,lb,meanll,varll] = mocap_hmsbn_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

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

A1 = parameters{12}; % 1*L
A2 = parameters{13}; % L*M

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+c)>rand(J,1));
end;


%% calculate lower bound
term1 = [binit,bsxfun(@plus,W1*h(:,1:T-1),b1)]; % J*T
logprior = sum(term1.*h-log(1+exp(term1))); % 1*T

mu = bsxfun(@plus,W2*h,b2); % M*T
logsigma = bsxfun(@plus,W3*h,b3); % M*T
loglike = -sum(logsigma + (v-mu).^2./(2*exp(2*logsigma))+1/2*log(2*pi)); % 1*T

term3 = [U2*v(:,1)+cinit, ...
    bsxfun(@plus,U1*h(:,1:T-1)+U2*v(:,2:T),c)]; % J*T
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

% disp(varll);

ll = (ll-meanll)./max(1,sqrt(varll));

%% gradient information
mat1 = h - sigmoid(term1); % J*T
grads.W1 = mat1(:,2:T)*h(:,1:T-1)'; % J*J
grads.b1 = sum(mat1(:,2:T),2); % J*1
grads.binit = mat1(:,1); % J*1

mat2 = (v-mu)./(exp(2*logsigma)); % M*T
grads.W2 = mat2*h'; % M*J
grads.b2 = sum(mat2,2); % M*1

mat2p = (v-mu).^2./(exp(2*logsigma))-1; % M*T
grads.W3 = mat2p*h'; % M*J
grads.b3 = sum(mat2p,2); % M*1

mat3 = bsxfun(@times,h-sigmoid(term3),ll); % J*T
grads.U1 = mat3(:,2:T)*h(:,1:T-1)'; % J*J
grads.U2 = mat3*v'; % J*M
grads.c = sum(mat3(:,2:T),2); % J*1 
grads.cinit = mat3(:,1);

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v';


%% collection
gradient{1} = grads.W1/(T-1);
gradient{2} = grads.W2/T;
gradient{3} = grads.W3/T;

gradient{4} = grads.U1/(T-1);
gradient{5} = grads.U2/T;

gradient{6} = grads.b1/(T-1);    
gradient{7} = grads.b2/T; 
gradient{8} = grads.b3/T; 
gradient{9} = grads.c/(T-1); 
gradient{10} = grads.binit; 
gradient{11} = grads.cinit; 

gradient{12} = grads.A1/T; 
gradient{13} = grads.A2/T; 

end        
