## How to use the code

The original mocap data can be downloaded from http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/

If you want to run cmu data, you need first download the preprocessed data from http://www-personal.umich.edu/~rmittelm/ (see the 
code link for the paper Structured RTRBM). We use this preprocessed dataset so that we can compare with their results directly.

After you downloaded it, you should be able to load tmp_MOCAP1.mat;  load tmp_MOCAP2.mat;

If you want to run mit data, the preprocessed data is already provided here. 

1. mocap_dtsbn_deter.m : a two-layer TSBN model with deterministic middle hidden layer;

2. mocap_dtsbn_stoc.m : a two-layer TSBN model with stochastic middle hidden layer;

3. mocap_hmsbn.m : a one-layer HMSBN model with order 1;

4. mocap_tsbn.m : a one-layer TSBN model with order 1;






 




