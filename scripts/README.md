This folder contains handy scripts for data pre-processing and plotting. 

----------------------------------------------------------------------------

This code was derived from the code produced by the Polavieja Lab and the Champalimaud Neuroscience Programme:
https://polavieja_lab.gitlab.io/

Their original "attention networks" code can be found here:
https://gitlab.com/polavieja_lab/fishandra

This code was published under a GPL license. Our code is an adaptation of the de Polavieja code enabling application to cell behavior. 

----------------------------------------------------------------------------

The Jupyter notebook "General_XML_Parser.ipynb" contains Python code demonstrating how to turn an XML file output from the ImageJ/FIJI TrackMate plugin into .npy files containing cell trajectory data. 

These .npy files are used to train and test the attention networks (see other folder for exact code we used for this), and to generate useful output files. 

Once the networks are trained/tested with output files generated, the following scripts may be used for plotting: 

The Jupyter notebook "General Cell Plotter with Voronoi.ipynb" contains Python code to generate snapshots of focal cells, their respective local neighbors, and the attention weights assigned to said neighbors, as in Fig. 1 of our paper. 

The Jupyter notebook "General_Attention_Heatmap_Plotter.ipynb" contains Python code to synthesize attention weights for all neighbors in a dataset, and generate the standard attention heatmaps and associated plots shown in our paper. 

The Jupyter notebook "Plot_HUVEC_Accuracies.ipynb" plots accuracy results for a sweep of trained networks. The .csv file "huvec_bulk.csv" is provided to demonstrate how accuracy results are pulled from it. Our train/test/output folder contains a handy script ("get_accs.py") for scraping accuracy results from a sweep of trained networks. 
