
function gradient = ar_delay_gradient(v,parameters)

W = parameters{1}; % M*M*nt
c  = parameters{2}; % M*1
[M,T] = size(v);
[~,~,nt] = size(W);

%% 
term = zeros(M,T); term(:,1) = c;

for t = 2:T
    cc = zeros(M,1);
    for delay = 1:min(t-1,nt)
        cc = cc + W(:,:,delay)*v(:,t-delay);
    end;
    term(:,t) = cc+c;
end;

%% gradient information
mat = v-sigmoid(term); % M*T
grads.W = zeros(M,M,nt);
for delay = 1:nt
    grads.W(:,:,delay) = mat(:,delay+1:T)*v(:,1:T-delay)'/(T-delay); % M*M
end;
grads.c = sum(mat,2)/T; % M*1

%% collection
gradient{1} = grads.W; 
gradient{2} = grads.c; 

end        
