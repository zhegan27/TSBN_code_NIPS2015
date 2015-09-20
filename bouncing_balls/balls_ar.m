clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for simulated data.
load bouncing_balls_training_data;
TrainData = Data;
N = length(TrainData);
[M,T]=size(TrainData{1}');

load bouncing_balls_testing_data;
TestData = Data;
Ntest = length(TestData);
clear Data;

%% Intialize parameters for training
initialParameters{1}=.001*randn(M,M); % W
initialParameters{2}=zeros(M,1); % c
initialParameters{3}=zeros(M,1); % cinit

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
[param,result]=ar_ascent(TrainData,initialParameters,opts,TestData);


