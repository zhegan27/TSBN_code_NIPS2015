function [v,N,ndx]=subsampleData(v,batchSize)
N=size(v,2); 
if batchSize==N
    v=v;
    ndx=1:N;
    return
end
ndx = datasample(1:N,batchSize,2,'Replace',false);
    v=v(:,ndx);
N=size(v,2);
end