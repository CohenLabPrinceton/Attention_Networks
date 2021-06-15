import os
import numpy as np
import pandas as pd

def data_config_summary(data_config):
    data_dict = {}
    mean_list = ['max_speed', 'mean_speed', 'median_speed', 'max_acceleration',
                 'median_acceleration', 'body_length', 'frames_per_second',
                 'num_animals']
    for key in mean_list:
        try:
            data_dict[key] = [np.mean([config[key] for config in data_config])]
        except TypeError:
            data_dict[key] = None

    for key in ['speed_percentiles', 'acceleration_percentiles']:
        data_dict[key] = [np.mean(np.asarray([config[key]
                                              for config in data_config]), axis=0)]

    for key in ['num_frames', 'num_train', 'num_validation', 'num_test']:
        data_dict[key] = [sum([config[key] for config in data_config])]

    data_dict['arena_radius'] = [np.mean([config["normalization_parameters"][-1]
                                          for config in data_config])]
    return data_dict

def size_maybe_list(possible_list):
    #TODO: This is temporary hack to accept history
    try:
        size = possible_list.shape[0]
    except AttributeError:
        size = [x.shape[0] for x in possible_list]
    return size

def save_train_results(var_dict):
    args = vars(var_dict['args'])
    results = {
               "num_train": size_maybe_list(var_dict['data']['train']),
               "num_validation": size_maybe_list(var_dict['data']['validation']),
               "num_test": size_maybe_list(var_dict['data']['test']),
               "val_loss_h": var_dict['history'].history['val_loss'],
               "val_acc_h": var_dict['history'].history['val_accuracy'],
               "train_loss_h": var_dict['history'].history['loss'],
               "train_acc_h": var_dict['history'].history['accuracy'],
               "learning_rate_h": var_dict['history'].history['lr'],
               "total_time": var_dict['end_train']-var_dict['start_train'],
               }
    for key in ["val_loss", "test_loss", "val_acc", "test_acc", "test_loss_ta",
                "test_loss_la", "test_acc_ta", "test_acc_la"]:
        results[key] = var_dict[key]

    results = {**results, **args}
    # args.update(results) #In case args overwrites something
    dict_of_lists = {key: [results[key]] for key in results}

    data_dict = data_config_summary(var_dict['datasets'].config)
    dict_of_lists.update(data_dict)
    results_df = pd.DataFrame(dict_of_lists)
    results_df.to_pickle(os.path.join(var_dict['args'].save_path, 'results.pkl'))
    return results_df
