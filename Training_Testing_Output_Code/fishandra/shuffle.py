import logging
import copy

import numpy as np

logger = logging.getLogger(__name__)


def shuffle_trajectories_in_place(trajectories, velocity, acceleration):
    logger.info("Shuffling trajectories in place")
    delay_per_ind = trajectories.shape[0] // trajectories.shape[1]
    # assert delay_per_ind * trajectories.shape[1] <= trajectories.shape[0]
    for i in range(1, trajectories.shape[1]):
        delay = delay_per_ind * i
        logger.debug("Shuffling {} with delay: {}".format(i, delay))
        for array in [trajectories, velocity, acceleration]:
            array[:,i] = np.concatenate([array[delay:, i],
                                         array[:delay, i]],
                                        axis = 0)
        if i%2 == 1: #Change sign of x (reflection) in odd individuals
            logger.debug("Refecting individual {}".format(i))
            for array in [trajectories, velocity, acceleration]:
                array[:, i, 0] = -array[:, i, 0]


def shuffle_social_context_in_place(social_context):
    logger.info("Shuffling social_context in place")
    shift = social_context.shape[0]//2
    social_context_copy = copy.deepcopy(social_context)
    social_context[:shift, : , 1:, :] = social_context_copy[-shift:, :, 1:, :]
    social_context[shift:, : , 1:, :] = social_context_copy[:-shift, :, 1:, :]
