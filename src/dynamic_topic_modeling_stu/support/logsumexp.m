function x=logsumexp(x)
c=max(x);
x=log(sum(exp(bsxfun(@minus,x,c))))+c;