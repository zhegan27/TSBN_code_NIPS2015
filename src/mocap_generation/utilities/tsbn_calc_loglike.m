
function lb = tsbn_calc_loglike(v,parameters)

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

end        

