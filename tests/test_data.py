import numpy as np
import penne
import penne.data as data
import penne.convert as convert
import tqdm


###############################################################################
# Test data.py
###############################################################################
def test_MDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.Dataset('MDB', partition)
        assert len(dataset[0]) == 2
        assert dataset[0][0].shape == (1, penne.WINDOW_SIZE)
        assert dataset[0][1].shape == (1,)

def test_PTDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.Dataset('PTDB', partition)
        assert len(dataset[0]) == 2
        assert dataset[0][0].shape == (1, penne.WINDOW_SIZE)
        assert dataset[0][1].shape == (1,)

def test_MDB_loader():
    batch_size = 3
    loader = data.loader('MDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets = next(it)
        assert features.shape == (batch_size, 1024)
        assert targets.shape == (batch_size,)

def test_PTDB_loader():
    batch_size = 3
    loader = data.loader('PTDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets = next(it)
        assert features.shape == (batch_size, 1024)
        assert targets.shape == (batch_size,)
