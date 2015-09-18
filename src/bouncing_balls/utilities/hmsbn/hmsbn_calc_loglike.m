
function lb = hmsbn_calc_loglike(v,parameters)

W1=parameters{1}; % J*J
W2=parameters{2}; % M*J
U1=parameters{3}; % J*J
U2=parameters{4}; % J*M
b=parameters{5}; % J*1
c=parameters{6}; % M*1
d=parameters{7}; % J*1
binit=parameters{8}; % J*1
dinit=parameters{9}; % J*1

[~,J] = size(W1); [~,T] = size(v);

%% feed-forward step
h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*v(:,1)+dinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*v(:,t)+d)>rand(J,1));
end;

%% calculate lower bound
term1 = [binit,bsxfun(@plus,W1*h(:,1:T-1),b)]; % J*T
term2 = bsxfun(@plus,W2*h,c); % M*T
term3 = [U2*v(:,1)+dinit, ...
    bsxfun(@plus,U1*h(:,1:T-1)+U2*v(:,2:T),d)]; % J*T

logprior = sum(term1.*h-log(1+exp(term1))); % 1*T
loglike = sum(term2.*v-log(1+exp(term2))); % 1*T
logpost = sum(term3.*h-log(1+exp(term3))); % 1*T
ll = logprior + loglike - logpost; % 1*T
lb = mean(ll);

end        

