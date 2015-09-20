clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for simulated data.
load bouncing_balls_training_data;
TrainData = Data;
N = length(TrainData);
[M,T]=size(TrainData{1}');
J=100; K = 100; nt = 2;

load bouncing_balls_testing_data;
TestData = Data;
Ntest = length(TestData);
clear Data;

%% Intialize parameters for training
initialParameters{1}=.001*randn(J,J,nt); % W1
initialParameters{2}=.001*randn(K,J); % W2
initialParameters{3}=.001*randn(J,K,nt); % W3
initialParameters{4}=.001*randn(K,K,nt); % W4
initialParameters{5}=.001*randn(M,K); % W5
initialParameters{6}=.001*randn(K,M,nt); % W6
initialParameters{7}=.001*randn(M,M,nt); % W7

initialParameters{8}=.001*randn(J,J,nt); % U1
initialParameters{9}=.001*randn(J,K); % U2
initialParameters{10}=.001*randn(J,K,nt); % U3
initialParameters{11}=.001*randn(K,K,nt); % U4
initialParameters{12}=.001*randn(K,M); % U5
initialParameters{13}=.001*randn(K,M,nt); % U6

initialParameters{14}=zeros(J,1); % b1
initialParameters{15}=zeros(K,1); % b2
initialParameters{16}=zeros(M,1); % b3
initialParameters{17}=zeros(J,1); % c1
initialParameters{18}=zeros(K,1); % c2

L = 100;
initialParameters{19}=.001*randn(1,L); % A1
initialParameters{20}=.001*randn(L,M); % A2


%% Training options

opts.iters=1e5; % iteration number
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
[param,result]=dtsbn_stoc_ascent(TrainData,initialParameters,opts,TestData);


