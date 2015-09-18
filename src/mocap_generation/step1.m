% Version 1.000 
%
% Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This is the "main" demo
% It trains two CRBM models, one on top of the other, and then
% demonstrates data generation

clear all; close all;
more off;   %turn off paging

%initialize RAND,RANDN to a different state
% rand('state',sum(100*clock))
% randn('state',sum(100*clock))

%Our important Motion routines are in a subdirectory
addpath('./Motion')

%Load the supplied training data
%Motion is a cell array containing 3 sequences of walking motion (120fps)
%skel is struct array which describes the person's joint hierarchy
load Data/data.mat

%Downsample (to 30 fps) simply by dropping frames
%We really should low-pass-filter before dropping frames
%See Matlab's DECIMATE function
dropframes;

fprintf(1,'Preprocessing data \n');

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%how-many timesteps do we look back for directed connections
%this is what we call the "order" of the model 
n1 = 3; %first layer
n2 = 3; %second layer
        
%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2
numdims = size(batchdata,2); %data (visible) dimension

%save some frames of our pre-processed data for later
%we need an initialization to generate 
initdata = batchdata(1:100,:);

numbatches = length(minibatch); 
traindata = cell(1,numbatches);

minibatch = cell(1,numbatches);
for batch = 1:numbatches-1
    minibatch{batch} = [(batch-1)*100+1:100*batch];
end;
minibatch{numbatches} = [3801:3826];

for batch = 1:numbatches,     
    numcases = length(minibatch{batch});
    mb = minibatch{batch}; %caches the indices   
    traindata{batch}= batchdata(mb,:);
end;

