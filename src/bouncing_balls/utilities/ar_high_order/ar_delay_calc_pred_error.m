
function pred_err = ar_delay_calc_pred_error(v,parameters)

W = parameters{1}; % M*M*nt
c  = parameters{2}; % M*1
[M,T] = size(v);
[~,~,nt] = size(W);

%% prediction
vpred = zeros(M,T); vpred(:,1) = sigmoid(c);

for t = 2:T
    cc = zeros(M,1);
    for delay = 1:min(t-1,nt)
        cc = cc + W(:,:,delay)*v(:,t-delay);
    end;
    vpred(:,t) = sigmoid(cc+c);
end;

pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));

end        

