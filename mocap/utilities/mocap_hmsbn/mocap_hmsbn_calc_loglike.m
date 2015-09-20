
function lb = mocap_hmsbn_calc_loglike(v,parameters)

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

end        

