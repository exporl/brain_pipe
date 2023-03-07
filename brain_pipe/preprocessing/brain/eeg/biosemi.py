"""Functions related to Biosemi EEG data."""
import logging

import numpy as np


def biosemi_trigger_processing_fn(trigger):
    """Process the trigger data from a Biosemi system.

    Parameters
    ----------
    trigger: np.ndarray
        The trigger data from a Biosemi system. The trigger data is expected to
        be a 1D array of integers.

    Returns
    -------
    np.ndarray
        The processed trigger data. The trigger data is a 1D array of integers.
    """
    triggers = trigger.flatten().astype(np.int32) & (2**16 - 1)
    values, counts = np.unique(triggers, return_counts=True)
    valid_mask = (0 < values) & (values < 256)
    val_indices = np.argsort(counts[valid_mask])
    most_common = values[valid_mask][val_indices[-1]]
    if triggers[0] != most_common:
        logging.warning("First value of the EEG triggers is on, shouldn't be the case")
    return np.int32(triggers != most_common)
