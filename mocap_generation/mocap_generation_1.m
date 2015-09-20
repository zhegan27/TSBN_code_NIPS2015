
% generate motions based on the parameters trained on Data/data.mat

close all
clc

rng( 25 )
 % walking : 20, 25
 % running : 20, 16 ,15
step1;

% clear all; clc; close all;
addpath( genpath('.') )

% Size and parameters for simulated data.
TrainData = traindata;
N = length(TrainData);
[M,~]=size(TrainData{1}');
J=100; nt = 3;

step3_gen;
step4_display;

