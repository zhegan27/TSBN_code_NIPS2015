
function lb = mocap_dtsbn_stoc_calc_loglike(v,parameters)

W1 = parameters{1}; % J*J*nt
W2 = parameters{2}; % K*J
W3 = parameters{3}; % J*K*nt
W4 = parameters{4}; % K*K*nt
W5 = parameters{5}; % M*K
W5prime = parameters{6}; % M*K
W6 = parameters{7}; % K*M*nt
W7 = parameters{8}; % M*M*nt
W7prime = parameters{9}; % M*M*nt

U1 = parameters{10}; % J*J*nt
U2 = parameters{11}; % J*K
U3 = parameters{12}; % J*K*nt
U4 = parameters{13}; % K*K*nt
U5 = parameters{14}; % K*M
U6 = parameters{15}; % K*M*nt

b1  = parameters{16}; % J*1
b2  = parameters{17}; % K*1
b3  = parameters{18}; % M*1
b3prime  = parameters{19}; % M*1
c1  = parameters{20}; % J*1
c2  = parameters{21}; % K*1

[~,J,nt] = size(W1); [K,~] = size(W2); [M,T] = size(v);

%% feed-forward step
h = zeros(K,T); z = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U5*v(:,1)+c2)>rand(K,1));
z(:,1) = double(sigmoid(U2*h(:,1)+c1)>rand(J,1));

for t = 2:T
    cc2 = zeros(K,1);  cc1 = zeros(J,1); 
    for delay = 1:min(t-1,nt) 
        cc2 = cc2 + U4(:,:,delay)*h(:,t-delay)+U6(:,:,delay)*v(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay)+U3(:,:,delay)*h(:,t-delay);
    end;
    h(:,t) = double(sigmoid(U5*v(:,t)+cc2+c2)>rand(K,1));
    z(:,t) = double(sigmoid(U2*h(:,t)+cc1+c1)>rand(J,1));
end;

%% calculate lower bound 
term1 = zeros(J,T); term1h = zeros(K,T);
mu = zeros(M,T); logsigma = zeros(M,T); 
term3h = zeros(K,T); term3 = zeros(J,T);
% sampling
term1(:,1) = b1;
term1h(:,1) = W2*z(:,1)+b2;
mu(:,1) = W5*h(:,1)+b3;
logsigma(:,1) = W5prime*h(:,1)+b3prime;
term3h(:,1) = U5*v(:,1)+c2;
term3(:,1) = U2*h(:,1)+c1;

for t = 2:T
    bb1 = zeros(J,1); bb2 = zeros(K,1); bb3 = zeros(M,1); 
    cc2 = zeros(K,1); cc1 = zeros(J,1); bb3prime = zeros(M,1);
    for delay = 1:min(t-1,nt)
        bb1 = bb1 + W1(:,:,delay)*z(:,t-delay)+W3(:,:,delay)*h(:,t-delay);
        bb2 = bb2 + W4(:,:,delay)*h(:,t-delay)+W6(:,:,delay)*v(:,t-delay);
        bb3 = bb3 + W7(:,:,delay)*v(:,t-delay);
        bb3prime = bb3prime + W7prime(:,:,delay)*v(:,t-delay);
        cc2 = cc2 + U4(:,:,delay)*h(:,t-delay)+U6(:,:,delay)*v(:,t-delay);
        cc1 = cc1 + U1(:,:,delay)*z(:,t-delay)+U3(:,:,delay)*h(:,t-delay);
    end;
    term1(:,t) = bb1+b1;
    term1h(:,t) = W2*z(:,t)+bb2+b2;
    mu(:,t) = W5*h(:,t)+bb3+b3;
    logsigma(:,t) = W5prime*h(:,t)+bb3prime+b3prime;
    term3h(:,t) = U5*v(:,t)+cc2+c2;
    term3(:,t) = U2*h(:,t)+cc1+c1;
end;

logprior = sum(term1.*z-log(1+exp(term1))) + ...
    sum(term1h.*h-log(1+exp(term1h))) ; % 1*T
loglike = -sum(logsigma + (v-mu).^2./(2*exp(2*logsigma))+1/2*log(2*pi)); % 1*T
logpost = sum(term3.*z-log(1+exp(term3))) +...
    sum(term3h.*h-log(1+exp(term3h))); % 1*T
ll = logprior + loglike - logpost; % 1*T
lb = mean(ll);

end        

