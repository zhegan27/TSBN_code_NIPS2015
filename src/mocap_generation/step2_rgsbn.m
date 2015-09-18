% clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for simulated data.
TrainData = traindata;
N = length(TrainData);
[M,~]=size(TrainData{1}');
J=100; nt = 3;

%% Intialize parameters for training
initialParameters{1}=.001*randn(J,J,nt); % W1
initialParameters{2}=.001*randn(M,J); % W2
initialParameters{3}=.001*randn(M,J); % W2prime
initialParameters{4}=.001*randn(J,M,nt)*0; % W3
initialParameters{5}=.001*randn(M,M,nt)*0; % W4
initialParameters{6}=.001*randn(M,M,nt)*0; % W4prime

initialParameters{7}=.001*randn(J,J,nt); % U1
initialParameters{8}=.001*randn(J,M); % U2
initialParameters{9}=.001*randn(J,M,nt)*0; % U3

initialParameters{10}=zeros(J,1); % b1
initialParameters{11}=zeros(M,1); % b2
initialParameters{12}=zeros(M,1); % b3
initialParameters{13}=zeros(J,1); % c

L = 100;
initialParameters{14}=.001*randn(1,L); % A1
initialParameters{15}=.001*randn(L,M); % A2

%% Training options

opts.iters=1e4; % iteration number
opts.penalties=1e-4; % weight decay
opts.decay=0; % learning rate decay
opts.momentum = 1; % 1: momentum is used 
opts.evalInterval=100;
opts.moment_val = 0.9;

% 0: SGD; 1: AdaGrad; 2: RMSprop
opts.method = 2;

opts.stepsize = 1e-4;
opts.rmsdecay = 0.95;

%%
[param,result]=rgsbn_ascent(TrainData,initialParameters,opts);

% save('mocap_HMSBN_order3_running.mat','param');
