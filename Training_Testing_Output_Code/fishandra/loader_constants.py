# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

# This module is used as a singleton
# This means that sometimes these variables will be modified
# particularly at the level of the script

# Each trajectory is divided in fractions (in the time dimension)
# Fractions with the same labels are concatenated
#FRACTIONS = (0.5, 0.01, 0.01, 0.01)
FRACTIONS = (0.5, 0.2, 0.2, 0.1)
LABELS = ("train", "validation", "test", "validation", "train")

# The datapoints where the focal fish is close to the border are removed
# We remove points that are outside a circle of REMOVE_OUTER fraction of
# the radius
REMOVE_OUTER = 0.8

# If SHUFFLE is not the empty string, different identities are shuffled
SHUFFLE = '' # 'trajectories' or 'social_context'


# Percentiles of speed saved. It used to be [5, 25, 75, 95]
PERCENTILES = tuple(range(0, 101))

# num of CPU used during the multi-processing steps
NUM_CPU = 4
MAX_BATCH_SIZE = 1024 #Default when doing predict (training b.s. much smaller)

# Use precalculated - removed 24/6/2019
# USE_PRECALCULATED = False
# SAVE_PRECALCULATED = False

#Constants that are actually variables worth adding in training dictionary
variables = ['FRACTIONS', 'LABELS', 'REMOVE_OUTER', 'SHUFFLE', 'PERCENTILES']

def cons_dictionary():
    return {key: globals()[key] for key in variables}
