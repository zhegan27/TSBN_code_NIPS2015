clear all; clc; close all;
addpath(genpath('.'))

%% load cmu data.
load tmp_MOCAP1.mat;
TrainData = Data;
N = length(TrainData);
[M,~]=size(TrainData{1}');
J=100;

load tmp_MOCAP2.mat;
TestData = Data;
Ntest = length(TestData);
clear Data;

%% load mit data
% load mit_mocap.mat;
% TrainData = train_data;
% N = length(TrainData);
% [M,~]=size(TrainData{1}');
% J=100;
% 
% TestData = test_data;
% Ntest = length(TestData);


%% Intialize parameters for training
initialParameters{1}=.001*randn(J,J); % W1
initialParameters{2}=.001*randn(M,J); % W2
initialParameters{3}=.001*randn(M,J); % W3

initialParameters{4}=.001*randn(J,J); % U1
initialParameters{5}=.001*randn(J,M); % U2

initialParameters{6}=zeros(J,1); % b1
initialParameters{7}=zeros(M,1); % b2
initialParameters{8}=zeros(M,1); % b3
initialParameters{9}=zeros(J,1); % c

initialParameters{10}=zeros(J,1); % binit
initialParameters{11}=zeros(J,1); % cinit

L = 100;
initialParameters{12}=.001*randn(1,L); % A1
initialParameters{13}=.001*randn(L,M); % A2

%% Training options

opts.iters=1e5; % iteration number
opts.penalties=1e-4; % weight decay
opts.decay=0; % learning rate decay
opts.momentum = 1; % 1: momentum is used 
opts.evalInterval=400;
opts.moment_val = 0.9;

% 0: SGD; 1: AdaGrad; 2: RMSprop
opts.method = 2;
opts.stepsize = 1e-4;
opts.rmsdecay = 0.95;

%%
[param,result] = mocap_hmsbn_ascent(TrainData,initialParameters,opts,TestData);
