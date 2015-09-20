## How to use the code

In order to generate visualized motions using our model, you need first go to http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/code.html, 
download mhmublv_code.zip, and put the code in this folder.

Now, you should be able to generate mocap motions by running mocap_generation_1.m or mocap_generation_2.m. The parameters learned
by training a order-3 HMSBN are provided in mocap_1.mat and mocap_2.mat.

mocap_1.mat is trained on Data/data.mat, and mocap_2.mat is trained on Data/ilya_running.mat.

You can also train the model by yourself by running mocap_training.m.





 




