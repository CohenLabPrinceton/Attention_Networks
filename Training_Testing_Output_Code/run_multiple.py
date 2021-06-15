import os 
import shutil

def run_and_output(train_folder, folder_name, num_ngbr, future_steps, num_epochs):
    
    conditions = 'NumNeighbors_' +  str(num_ngbr)
    conditions += '_FutureSteps_' + str(future_steps)
    conditions += '_NumEpochs_' + str(num_epochs)
    conditions += '_RemoveOuter'
    
    folder_name_w_conditions = '/tigress/jl40/save_june/' + folder_name + '_' + conditions
    if not os.path.exists(folder_name_w_conditions):
        os.makedirs(folder_name_w_conditions)
    
    # Train the network and also produce the stupid output file and stuff 
 
    print('Training......')

    command = 'python ./fastrain '+train_folder+'*.npy --save_path '+ folder_name_w_conditions +' --num_neighbours '+str(num_ngbr)+' --future_steps '+str(future_steps)+' --num_epochs '+str(num_epochs) + ' --batch_size 1000'
    print(command)
    
    os.system(command)
    
    new_results = folder_name_w_conditions + '/results_train.log'
    new_debug = folder_name_w_conditions + '/debug_train.log'
    shutil.move('./logs/results_huvec_2.log', new_results)
    shutil.move('./logs/debug_huvec_2.log', new_debug)
    
    print('Testing......')
    
    command = 'python ./fastest '+train_folder+'*.npy --model_folder '+ folder_name_w_conditions + ' --future_steps ' + str(future_steps)
    print(command)
    
    os.system(command)
    
    
    new_results = folder_name_w_conditions + '/results_test.log'
    new_debug = folder_name_w_conditions + '/debug_test.log'
    
    shutil.move('./logs/results_huvec_2.log', new_results)
    shutil.move('./logs/debug_huvec_2.log', new_debug)
    
    print('Generating output......')
    
    command = 'python output_results.py ' + folder_name_w_conditions
    print(command)
    
    os.system(command)
    
    print('Done.')
    print('-------------------------------------------------------------------------')
    
# ------------------------------------------------------------------------------------------------------ #

list_subfolders_with_paths = [f.path for f in os.scandir('./NPYs') if f.is_dir()]
list_subfolders_with_paths.sort()

for i in range(len(list_subfolders_with_paths)):
    
    # Print the subfolder name: 
    #print(list_subfolders_with_paths[i])
    train_folder = list_subfolders_with_paths[i] + '/'
    #test_folder = list_subfolders_with_paths[i] + '/test/'
    
    # Generate the output folder with [the same folder name] + [run conditions]
    # Strip first bit off subfolder name:
    folder_name = list_subfolders_with_paths[i][7:]
    print(folder_name)
    
    num_epochs = 1000

    for j in range(5,55,5):
        num_ngbr = j
        
        future_steps = 2
        run_and_output(train_folder, folder_name, num_ngbr, future_steps, num_epochs)
        
        future_steps = 4
        run_and_output(train_folder, folder_name, num_ngbr, future_steps, num_epochs)
