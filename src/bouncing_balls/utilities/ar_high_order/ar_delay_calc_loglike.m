
function lb = ar_delay_calc_loglike(v,parameters)

W = parameters{1}; % M*M*nt
c  = parameters{2}; % M*1
[M,T] = size(v);
[~,~,nt] = size(W);

%% calculate log likelihood 
term = zeros(M,T); term(:,1) = c;

for t = 2:T
    cc = zeros(M,1); 
    for delay = 1:min(t-1,nt)
        cc = cc + W(:,:,delay)*v(:,t-delay);
    end;
    term(:,t) = cc+c;
end;

loglike = sum(term.*v-log(1+exp(term))); % 1*T
lb = mean(loglike);

end        

