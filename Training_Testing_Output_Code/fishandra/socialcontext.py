from sklearn.neighbors import NearestNeighbors
import numpy as np

def context_content(neighbours, focal):
    """Gives context content of the neighbours
    New position is relative position.
    Velocity and acceleration unchanged
    """
    context = neighbours.copy()
    context[:, :, :2] -= focal[:, np.newaxis, :2]
    return context

def context_asocial_content(focal):
    """Gives context content for the focal.
    Everything unchanged, but position relative to itself is 0"""
    context = focal.copy()
    context[..., :2] = 0
    return context

def context_h_content(history):
    """Gives context content of the history for both focal and neighbour wrt
    to the present position of the focal"""
    # Untested !!!!
    context = history.copy()
    # Substract last location of focal from ALL coordinates
    context -= context[:,0,-1][:,np.newaxis,np.newaxis,...]
    return context

def give_me_social_context(data, num_neighbours, history = None):
    """Given some trajectories (data with velocities and accelerations) for
    several individuals it returns an array with the relavite positions of the
    num_neighbours-nearest neighbours and the absolute velocities and accelerations
    for the focal and neighbours. Note that the position of the focal wrt. itself
    is 0.

    Parameters
    ----------

    data : ndarray
        Array of shape (frames, individuals, coordinates) with animal trajectories
    num_neighbours : int
        Number of neighbours to consider for the social context

    Returns
    -------
    context : ndarray
        Array of shape (frames, individuals, num_neighbours + 1, coordinates)
        with the relative positions (maybe also absolute velocities and acceleration) of the
        num_neighbours neighbours for every frame. The positions of the focal
        (first individual in the array) are set to 0.
    context_history : ndarray
        Array of shape (frames, individuals, num_neighbours + 1, history_steps,
        2)
    """
    [total_time_steps, individuals, coordinates] = data.shape
    # Coordinates will normally be 6: pos, vel and acc
    context = np.empty([total_time_steps, individuals,
                        num_neighbours + 1, coordinates], dtype=data.dtype)
    neighbours_indices = np.empty([total_time_steps, individuals,
                                   num_neighbours + 1], dtype=data.dtype)

    if history is not None:
        [total_time_steps_h, individuals_h, history_steps, coordinates_h] = history.shape
        assert total_time_steps_h == total_time_steps
        assert individuals_h == individuals
        assert coordinates_h == 2
        # Coordinates of history will normally be two (relative x, y)
        context_history = np.empty([total_time_steps, individuals,
                                    num_neighbours + 1, history_steps,
                                    coordinates_h], dtype=data.dtype)
    else:
        context_history = None

    for frame in range(total_time_steps):
        positions = data[frame, :, :2]
        nbrs = NearestNeighbors(n_neighbors=num_neighbours+1,
                                algorithm='ball_tree').fit(positions)
        indices = nbrs.kneighbors(positions, return_distance=False)
        context[frame,:,1:, :] = context_content(data[frame, indices[:,1:] ,:],
                                                 data[frame,...])
        context[frame,:,0,:] = context_asocial_content(data[frame,...])
        neighbours_indices[frame,:,:] = indices

        if history is not None:
            context_history[frame, ...] = context_h_content(history[frame, indices[:,:], :, :])

    return context, context_history, neighbours_indices
