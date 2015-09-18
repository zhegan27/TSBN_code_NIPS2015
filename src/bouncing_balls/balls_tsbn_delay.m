clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for simulated data.
load bouncing_balls_training_data;
TrainData = Data;
N = length(TrainData);
[M,T]=size(TrainData{1}');
J=100; % number of hidden units 
nt = 2; % the order of the model

load bouncing_balls_testing_data;
TestData = Data;
Ntest = length(TestData);
clear Data;

%% Intialize parameters for training
initialParameters{1}=.001*randn(J,J,nt); % W1
initialParameters{2}=.001*randn(M,J); % W2
initialParameters{3}=.001*randn(J,M,nt); % W3
initialParameters{4}=.001*randn(M,M,nt); % W4

initialParameters{5}=.001*randn(J,J,nt); % U1
initialParameters{6}=.001*randn(J,M); % U2
initialParameters{7}=.001*randn(J,M,nt); % U3

initialParameters{8}=zeros(J,1); % b
initialParameters{9}=zeros(M,1); % c
initialParameters{10}=zeros(J,1); % d

% this is the parameters that are used to learn the data-dependent baseline
% using a feed-forward neural network
L = 100;
initialParameters{11}=.001*randn(1,L); % A1
initialParameters{12}=.001*randn(L,M); % A2

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
[param,result]=tsbn_delay_ascent(TrainData,initialParameters,opts,TestData);
