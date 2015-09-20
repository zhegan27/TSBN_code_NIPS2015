function [Topics]=OutputTopics(WP,WO,W_outputN)
%W_outputN: output terms for each topic
% WP_sum=sum(WP);%total number of terms assigned to each topic
[~,Z]=size(WP);%T: # of terms;Z: # of topics
Topics=cell(Z,1);
for z=1:Z
    [~,TermIndex]=sort(WP(:,z),'descend');
    str='';
    for index=1:W_outputN       
        str=[str ' ' WO{TermIndex(index)}];
    end
    Topics{z}=str;
end
