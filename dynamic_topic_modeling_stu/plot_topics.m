clear all; clc; close all;
addpath(genpath('.'));

% Step 1, load stu datga
rand('seed', 1); randn('seed', 1); binornd('seed', 1);

load('STOU.mat');
WCmatrix = STOU.WCmatrix;
[nV, nT] = size(WCmatrix);
minl = 5; ming = 20;  voc = STOU.words;
% 
if 0
minl = 5; ming = 20; 
ind1 = find( sum (WCmatrix,2 ) < ming)';
[nV, nT] = size(WCmatrix);

ind2 = [];
for v = 1:nV
    if max ( WCmatrix(v,:) ) < minl  
        ind2 = [ind2, v];
    end
end
WCmatrix = WCmatrix(setdiff(1:nV,[ind1,ind2]), :); 
end

load('dtsbn_200_3_10_b.mat');

WP = param{2}; W_outputN = 10;
Topics = OutputTopics(WP,voc,W_outputN);



TestData = WCmatrix(:,nT); tmpTrainData =  WCmatrix(:,1:end-1); [nV ,nT_tr] = size(tmpTrainData);
% hold out 20% for testing % index = randperm(N);
hd = 0.2;  t_total = sum(tmpTrainData,1)'; nhd = ceil(hd*t_total);
TrainData = zeros(nV, nT_tr); HdoutData = zeros(nV, nT_tr);
for t = 1:nT_tr
    doc = [];
    for v = 1:nV
        if tmpTrainData(v,t)~= 0
            doc = [doc, v*ones(1,tmpTrainData(v,t))];
        end
    end
    hd_index = randperm(t_total(t));
    hd_doc = doc( hd_index(1:nhd(t) )); tr_doc = doc( hd_index(nhd(t)+1: end )); 
    
    for v = 1:nV 
        HdoutData(v, t) =  length(find(hd_doc == v));
        TrainData(v, t) =  length(find(tr_doc == v));
    end
end

%% Size and parameters for simulated data.
[M,T]=size(TrainData); [~,Ntest] = size(TestData); J=200; L = 200;

U1=param{3}; % J*J
U2=param{4}; % J*M
b2=param{6}; % M*1
c=param{7}; % J*1
cinit=param{9}; % J*1

h = zeros(J,T);
% sampling
h(:,1) = double(sigmoid(U2*TrainData(:,1)+cinit)>rand(J,1));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*TrainData(:,t)+c)>rand(J,1));
end;

h(:,1) = double(sigmoid(U2*TrainData(:,1)+cinit));
for t = 2:T
    h(:,t) = double(sigmoid(U1*h(:,t-1)+U2*TrainData(:,t)+c));
end;

nl = 10;[~,min_id] = sort(sum(h,2)/T); 
yy = h(min_id(1:nl) ,:);
styles = {'g<-','r<-','b<-','k<-'};
for i = 1:nl
    figure(i);
    axes1 = axes('FontSize', 20); 
    xx = 1790: 1790+T-1; 
    plot(xx,yy(i,:),styles{1},'LineWidth',2,'MarkerSize',3); hold on;
end
xlabel('year'); ylabel('usage');

id_topics = [29, 30, 130];
yy = h(id_topics,:);
styles = {'g<-','r<-','b<-','k<-'};
for i = 1:3
    figure(i);
    usage = yy(i,:);
    usage = (usage - min(usage)) / (max(usage) - min(usage))
    xx = 1790: 1790+T-1; 
    plot(xx,usage,styles{i},'LineWidth',2,'MarkerSize',3);
    set(gca,'xtick',xx(1:80:224),'xticklabel',num2str(xx(1:80:224)) ,'FontSize', 20);
    legend(['Topic ', num2str(id_topics(i))],'FontSize', 20); grid on;
end

save('dtsbn_topics.mat', 'Topics', 'h');

save_name = ['dtsbn_', num2str(J), '.mat'];
save(save_name , 'param', 'result');




