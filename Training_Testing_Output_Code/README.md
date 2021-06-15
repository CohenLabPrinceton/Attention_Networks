This is the exact code we used to train and test the deep attention networks from cell trajectory data, and to output results (e.g. accuracy data) into a useful format. 

The Princeton TigerGPU cluster was used to train individual networks. Princeton researchers may use the train_multiple.cmd file in this cluster environment to run the network training, testing, and output generation sweeps as was done in our paper. 

For more information on TigerGPU, please reference: 
https://researchcomputing.princeton.edu/systems/tiger

This command script runs the Python file named "run_multiple.py". This must be given a path to your subfolders containing .npy files of cell trajectory data, as well as an output path for saving models and results. 
