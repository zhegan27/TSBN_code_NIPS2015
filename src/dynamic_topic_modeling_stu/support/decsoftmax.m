function x=decsoftmax(x)

x=exp(bsxfun(@minus,x,logsumexp(x)));