"""
    dataset util class
"""
import numpy as np


class DatasetIndexer(object):
    """Utility class for mapping given indices to provided dataset.
    Parameters
    ----------
    dataset : `torch.utils.data.Dataset`
    """
    def __init__(self, dataset, ind):
        self.dataset = dataset
        self.ind = np.asarray(ind)
        assert ind.min() >= 0 and ind.max() <= len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[self.ind[index]]

    def __len__(self):
        return len(self.ind)
