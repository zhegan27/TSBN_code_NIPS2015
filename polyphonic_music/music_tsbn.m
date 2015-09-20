clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for simulated data.
load MuseData;
% load JSB_Chorales;
% load Nottingham;
% load Piano_midi;

TrainData = traindata;
N = length(TrainData);
[M,~]=size(TrainData{1}');
J=100; %nt = 4;

TestData = testdata;
Ntest = length(TestData);
clear traindata testdata validdata;

%% Intialize parameters for training
initialParameters{1}=.001*randn(J,J); % W1
initialParameters{2}=.001*randn(M,J); % W2
initialParameters{3}=.001*randn(J,M); % W3
initialParameters{4}=.001*randn(M,M); % W4

initialParameters{5}=.001*randn(J,J); % U1
initialParameters{6}=.001*randn(J,M); % U2
initialParameters{7}=.001*randn(J,M); % U3

initialParameters{8}=zeros(J,1); % b
initialParameters{9}=zeros(M,1); % c
initialParameters{10}=zeros(J,1); % d

initialParameters{11}=zeros(J,1); % b
initialParameters{12}=zeros(M,1); % c
initialParameters{13}=zeros(J,1); % d

L = 100;
initialParameters{14}=.001*randn(1,L); % A1
initialParameters{15}=.001*randn(L,M); % A2

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
[param,result] = tsbn_ascent(TrainData,initialParameters,opts,TestData);

