
function lb = tsbn_delay_calc_loglike(v,parameters)

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

end        

