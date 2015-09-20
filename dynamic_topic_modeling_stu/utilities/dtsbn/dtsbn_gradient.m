
function [gradient,lb,meanll,varll] = dtsbn_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
U1=parameters{3}; % J*J
U2=parameters{4}; % J*M
b1=parameters{5}; % J*1
b2=parameters{6}; % M*1
c=parameters{7}; % J*1
binit=parameters{8}; % J*1
cinit=parameters{9}; % J*1

A1 = parameters{10}; % 1*L
A2 = parameters{11}; % L*M

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
term2 = bsxfun(@plus,W2*h,b2); % M*T
term3 = [U2*v(:,1)+cinit, ...
    bsxfun(@plus,U1*h(:,1:T-1)+U2*v(:,2:T),c)]; % J*T

ttotal = sum(v,1);
logprior = sum(term1.*h-log(1+exp(term1))); % 1*T
loglike = sum(term2.*v-bsxfun(@times,log(logsumexp(term2)), v )); % 1*T
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
grads.b1 = sum(mat1(:,2:T),2); % J*1
grads.binit = mat1(:,1); % J*1
% grads.b1 = zeros( size( sum(mat1(:,2:T),2) )); % J*1
% grads.binit = zeros( size( mat1(:,1) )); % J*1


mat2 = v - bsxfun(@times,ttotal, decsoftmax(term2)); % M*T
grads.W2 = mat2*h' ; % M*J
grads.b2 = sum(mat2,2); % M*1
% grads.b2 = zeros( size( sum(mat2,2) )); % M*1

mat3 = bsxfun(@times,h-sigmoid(term3),ll); % J*T
grads.U1 = mat3(:,2:T)*h(:,1:T-1)'; % J*J
grads.U2 = mat3*v'; % J*M
grads.c = sum(mat3(:,2:T),2); % J*1 
grads.cinit = mat3(:,1);

% grads.c = zeros( size( sum(mat3(:,2:T),2) )); % J*1 
% grads.cinit = zeros( size( mat3(:,1) ));


grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v';

%% collection
gradient{1} = grads.W1/(T-1);
gradient{2} = grads.W2/T;
gradient{3} = grads.U1/(T-1);
gradient{4} = grads.U2/T;
gradient{5} = grads.b1/(T-1);    
gradient{6} = grads.b2/T; 
gradient{7} = grads.c/(T-1); 
gradient{8} = grads.binit; 
gradient{9} = grads.cinit; 

gradient{10} = grads.A1/T; 
gradient{11} = grads.A2/T; 

end        
