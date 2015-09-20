
% generate motions based on the parameters trained on Data/ilya_running.mat

clear all
close all
clc

 rng( 39 ) 
% rand('seed', 22)

% running 20 , 19 , 22, 3? 14
step1_running;

% clear all; clc; close all;
addpath( genpath('.') )

% Size and parameters for simulated data.
TrainData = traindata;
N = length(TrainData);
[M,~]=size(TrainData{1}');
J=100; nt = 3;

step3_gen_running;
step4_display;

