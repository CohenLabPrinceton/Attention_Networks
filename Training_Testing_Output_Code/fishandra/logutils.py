# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import logging

def createlogs(results, debug): 

    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(debug)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatterch = logging.Formatter('%(message)s')
    #ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    ch.setFormatter(formatterch)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logresults = logging.getLogger('results')
    logresults.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(results)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level ## Remove, as already done above
    #ch2 = logging.StreamHandler()
    #ch2.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    #ch2.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    #logresults.addHandler(ch2)
    logresults.addHandler(fh)

    return logresults, logger
