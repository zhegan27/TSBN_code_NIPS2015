
function lb = ar_calc_loglike(v,parameters)

W = parameters{1}; % M*M
c  = parameters{2}; % M*1
cinit = parameters{3}; % M*1
[~,T] = size(v);


%% calculate log likelihood
term = [cinit,bsxfun(@plus,W*v(:,1:T-1),c)]; % M*T
loglike = sum(term.*v-log(1+exp(term))); % 1*T
lb = mean(loglike);

end        

