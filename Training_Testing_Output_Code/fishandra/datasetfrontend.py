""" datasetfrontend

Tools to select variables to be used during prediction. It works by blinding
(that is: rewritting by 0) all the variables mentioned during the instantiation
Currently used in loader.py
"""

import copy
import logging
logger = logging.getLogger(__name__)

class BlindingFrontEnd():
    """A front end that blinds variable from the key social"""
    all_options = ["fv", "nbv", "fa", "nba", "fat",
                   "nbat", "fan", "nban", "nb"]
    options_full_exp = {"fv": "focal v", "nbv": "neighbour v",
                        "fa": "focal a", "nba": "neighbour a",
                        "fan": "focal a normal", "nban": "neighbour a normal",
                        "fat": "focal a tg", "nbat": "neighbour a tg",
                        "nb": "no social info"}

    def __init__(self, blind=[]):
        self.blind = {}
        for key in self.all_options:
            if key in blind:
                logger.debug("Blinding " + self.options_full_exp[key])
                self.blind[key] = True
            else:
                self.blind[key] = False

    def __call__(self, batch):
        batch['social'] = copy.deepcopy(batch['social'])
        if self.blind['fv']:
            batch['social'][..., 0, 2:4] = 0
        if self.blind['nbv']:
            batch['social'][..., 1:, 2:4] = 0
        if self.blind['fat']:
            batch['social'][..., 0, 5] = 0
        if self.blind['nbat']:
            batch['social'][..., 1:, 5] = 0
        if self.blind['fan']:
            batch['social'][..., 0, 4] = 0
        if self.blind['nban']:
            batch['social'][..., 1:, 4] = 0
        if self.blind['fa']:
            batch['social'][..., 0, 4:6] = 0
        if self.blind['nba']:
            batch['social'][..., 1:, 4:6] = 0
        if self.blind['nb']:
            batch['social'][..., 1:, :] = 0
        return batch

class NoFrontEnd():
    """ A dummy frontend that blinds no variable"""
    def __init__(self):
        pass
    def __call__(self, batch):
        return batch
