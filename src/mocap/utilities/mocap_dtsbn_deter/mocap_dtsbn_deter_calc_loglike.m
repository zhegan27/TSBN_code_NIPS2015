
function lb = mocap_dtsbn_deter_calc_loglike(v,parameters)

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

end        

