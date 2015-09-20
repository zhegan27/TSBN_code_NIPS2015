
function [gradient,lb,meanll,varll] = tsbn_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

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

A1 = parameters{14}; % 1*L
A2 = parameters{15}; % L*M

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+dinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+U3*v(:,t-1)+d)>rand(J,1));
end;

%% calculate lower bound
term1 = [binit,bsxfun(@plus,W1*h(:,1:T-1)+W3*v(:,1:T-1),b)]; % J*T
term2 = [bsxfun(@plus,W2*h(:,1),cinit), ...
    bsxfun(@plus,W2*h(:,2:T)+W4*v(:,1:T-1),c)]; % M*T
term3 = [U2*v(:,1)+dinit, ...
    bsxfun(@plus,U1*h(:,1:T-1)+U2*v(:,2:T)+U3*v(:,1:T-1),d)]; % J*T

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
grads.W1 = mat1(:,2:T)*h(:,1:T-1)'; % J*J
grads.W3 = mat1(:,2:T)*v(:,1:T-1)'; % J*M
grads.b = sum(mat1(:,2:T),2); % J*1
grads.binit = mat1(:,1); % J*1

mat2 = v-sigmoid(term2); % M*T
grads.W2 = mat2*h' ; % M*J
grads.W4 = mat2(:,2:T)*v(:,1:T-1)' ; % M*M
grads.c = sum(mat2(:,2:T),2); % M*1
grads.cinit = mat2(:,1); % M*1

mat3 = bsxfun(@times,h-sigmoid(term3),ll); % J*T
grads.U2 = mat3*v'; % J*M
grads.U1 = mat3(:,2:T)*h(:,1:T-1)'; % J*J
grads.U3 = mat3(:,2:T)*v(:,1:T-1)'; % J*M
grads.d = sum(mat3(:,2:T),2); % J*1 
grads.dinit = mat3(:,1);

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v';

%% collection
gradient{1} = grads.W1/(T-1);
gradient{2} = grads.W2/T;
gradient{3} = grads.W3/(T-1);
gradient{4} = grads.W4/(T-1);

gradient{5} = grads.U1/(T-1);
gradient{6} = grads.U2/T;
gradient{7} = grads.U3/(T-1);

gradient{8} = grads.b/(T-1);    
gradient{9} = grads.c/(T-1); 
gradient{10} = grads.d/(T-1); 
gradient{11} = grads.binit; 
gradient{12} = grads.cinit; 
gradient{13} = grads.dinit; 

gradient{14} = grads.A1/T; 
gradient{15} = grads.A2/T; 

end        
