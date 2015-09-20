clear all; clc; close all;
addpath(genpath('.'));

% Step 1, load stu datga
rand('seed', 1); randn('seed', 1); binornd('seed', 1);

load('STOU.mat');
WCmatrix = STOU.WCmatrix;
[nV, nT] = size(WCmatrix);
minl = 3; ming = 10; 

if 1
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

%% Intialize parameters for training
[M,T]=size(TrainData); [~,Ntest] = size(TestData); J=25; K=25; L = 25; nt =1;

initialParameters{1}=.001*randn(J,J,nt); % W1
initialParameters{2}=.001*randn(K,J); % W2
initialParameters{3}=0*randn(J,K,nt); % W3
initialParameters{4}=.001*randn(K,K,nt); % W4
initialParameters{5}=.001*randn(M,K); % W5
initialParameters{6}=0*randn(K,M,nt); % W6
initialParameters{7}=0*randn(M,M,nt); % W7

initialParameters{8}=.001*randn(J,J,nt); % U1
initialParameters{9}=.001*randn(J,K); % U2
initialParameters{10}=0*randn(J,K,nt); % U3
initialParameters{11}=.001*randn(K,K,nt); % U4
initialParameters{12}=.001*randn(K,M); % U5
initialParameters{13}=0*randn(K,M,nt); % U6

initialParameters{14}=zeros(J,1); % b1
initialParameters{15}=zeros(K,1); % b2
initialParameters{16}=zeros(M,1); % b3
initialParameters{17}=zeros(J,1); % c1
initialParameters{18}=zeros(K,1); % c2

initialParameters{19}=.001*randn(1,L); % A1
initialParameters{20}=.001*randn(L,M); % A2

%% Training options

opts.iters=5e3; % iteration number
opts.penalties=1e-4; % weight decay
opts.decay=0; % learning rate decay
opts.momentum = 1; % 1: momentum is used 
opts.evalInterval=10;
opts.moment_val = 0.9;

% 0: SGD; 1: AdaGrad; 2: RMSprop
opts.method = 2;

opts.stepsize = 1e-4;
opts.rmsdecay = 0.95;

%%
[param,result]=ddtsbn_stoc_ascent(TrainData,initialParameters,opts,HdoutData,TestData);

save_name = ['ddtsbn_', num2str(J),'_', num2str(minl),'_0','.mat'];
save(save_name , 'param', 'result');

