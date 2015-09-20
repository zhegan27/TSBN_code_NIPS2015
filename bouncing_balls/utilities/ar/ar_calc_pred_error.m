
function pred_err = ar_calc_pred_error(v,parameters)

W = parameters{1}; % M*M
c  = parameters{2}; % M*1
cinit = parameters{3}; % M*1
[~,T] = size(v);

%% prediction
vpred = [sigmoid(cinit),sigmoid(bsxfun(@plus,W*v(:,1:T-1),c))];
pred_err = mean(sum((v(:,2:T)-vpred(:,2:T)).^2));

end        

