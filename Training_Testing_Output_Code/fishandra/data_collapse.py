""" data_collapse

Tools that rotate each scene in a frame of reference that it is
at rest but shifted so each focal fish is placed in 0,0, and rotated in such a
way that the focal fish velocity extends to the positive y semiaxis.
"""

import logging
import numpy as np
import trajectorytools as tt

logger = logging.getLogger(__name__)

def collapse_one(rotation_matrix, data):
    if len(data.shape) == 3:
        return np.einsum('mlij,mlj->mli', rotation_matrix, data)
    elif len(data.shape) == 4:
        return np.einsum('mlij,mlkj->mlki', rotation_matrix, data)
    elif len(data.shape) == 5: #This is history
        return np.einsum('mlij,mlkhj->mlkhi', rotation_matrix, data)
    else:
        raise Exception

def collapse(rot, data, collapsable_pairs = None):
    logger.debug("Collapsable pairs" + str(collapsable_pairs))
    if collapsable_pairs is None:
        max_collapsable = data.shape[-1]
    else:
        max_collapsable = collapsable_pairs*2
    for i in range(0,max_collapsable,2):
        data[...,i:i+2] = collapse_one(rot, data[...,i:i+2])

def collapse_data_by_rotating(all_data, collapsable = {}):
    """collapse_data_by_rotating

    :param all_data: Trajectories to be modified in place
    :param collapsable: A dictionary indicating how many pairs of variables
    are to be collapsed.
    """
    logger.debug("Collapsing data...")
    logger.debug("Collapsable", collapsable)
    # Social rotation matrix: brings velocity to the positive y axis
    social_rotation_matrix = tt.matrix_rotate_to_vector(all_data['asocial'][...,2:4])
    # Asocial rotation matrix: brings position to the positive y axis
    rotation_matrix = tt.matrix_rotate_to_vector(all_data['asocial'][...,:2])
    # Target, history and social context comove with focal velocity
    collapse(social_rotation_matrix, all_data['target'],
            collapsable_pairs = collapsable.get('target'))
    collapse(social_rotation_matrix, all_data['social'],
            collapsable_pairs = collapsable.get('social'))
    if 'history' in all_data.keys():
        collapse(social_rotation_matrix, all_data['history'],
                collapsable_pairs = collapsable.get('history'))
    # Asocial is collapsed so position is in the positive y axis
    collapse(rotation_matrix, all_data['asocial'],
            collapsable_pairs = collapsable.get('asocial'))
    return rotation_matrix, social_rotation_matrix


# COLLAPSE MODULE
#
#class CollapseModuleMetricModel():
#    def __call__(self, all_data): #Might need to add collapsable
#        _, self.social_rotation_matrix = collapse_data_by_rotating(all_data)
#    def undo(self, result):
#        uncollapse(self.social_rotation_matrix, result)
#        return result
#    def __init__(self):
#        pass
#
#def uncollapse(rotation_matrix, data):
#    collapse(rotation_matrix.transpose((0,1,3,2)), data)
#    return data
