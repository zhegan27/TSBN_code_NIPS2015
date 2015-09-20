
function gradient = ar_gradient(v,parameters)

W = parameters{1}; % M*M
c  = parameters{2}; % M*1
cinit = parameters{3}; % M*1
[~,T] = size(v);

%% gradient information
term = [cinit,bsxfun(@plus,W*v(:,1:T-1),c)]; % M*T
mat = v-sigmoid(term); % M*T
grads.W = mat(:,2:T)*v(:,1:T-1)' ; % M*M
grads.c = sum(mat(:,2:T),2); % M*1
grads.cinit = mat(:,1); % M*1

%% collection
gradient{1} = grads.W/(T-1);
gradient{2} = grads.c/(T-1);
gradient{3} = grads.cinit; 

end        
