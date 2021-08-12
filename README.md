# Attention_Networks
Our code for the application of deep attention networks to recover collective rules from cell trajectory data.

This code was derived from the code produced by the Polavieja Lab and the Champalimaud Neuroscience Programme:
https://polavieja_lab.gitlab.io/

Their original "attention networks" code can be found here:
https://gitlab.com/polavieja_lab/fishandra

This code was published under a GPL license. Our code is an adaptation of the de Polavieja code enabling application to cell behavior. 

----------------------------------------------------------------------------

The folder "Training_Testing_Output_Code" contains the code we utilized (via the Princeton TigerGPU cluster) to train and test networks and generate output in a useful format. 

The folder "scripts" contains the code we used to pre-process data (from TrackMate XML files to Python .npy files) and plot results (e.g., generating the attention heatmaps and accuracy plots as can be seen in our paper). 

The folder "All_Accuracy_Data" contains all the accuracy results in .csv format for experiments described in the paper. 
