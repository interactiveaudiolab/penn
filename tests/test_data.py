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

###############################################################################
# NVD Tests
###############################################################################

def test_nvd_MDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.NVDDataset('MDB', partition)
        assert len(dataset[0]) == 2
        assert dataset[0][0].shape[0] == penne.PITCH_BINS
        assert dataset[0][1].shape[0] == 2
        assert dataset[0][0].shape[1] == dataset[0][1].shape[1]

def test_nvd_PTDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.NVDDataset('PTDB', partition)
        assert len(dataset[0]) == 2
        assert dataset[0][0].shape[0] == penne.PITCH_BINS
        assert dataset[0][1].shape[0] == 2
        assert dataset[0][0].shape[1] == dataset[0][1].shape[1]

def test_nvd_MDB_loader():
    batch_size = 3
    loader = data.nvd_loader('MDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets, ar = next(it)
        assert features.shape == (batch_size, penne.PITCH_BINS, 1)
        assert targets.shape == (batch_size, 2, 1)
        assert ar.shape == (batch_size, 2, 100)

def test_nvd_PTDB_loader():
    batch_size = 3
    loader = data.nvd_loader('PTDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets, ar = next(it)
        assert features.shape == (batch_size, penne.PITCH_BINS, 1)
        assert targets.shape == (batch_size, 2, 1)
        assert ar.shape == (batch_size, 2, 100)

###############################################################################
# AR Tests
###############################################################################

def test_ar_MDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.ARDataset('MDB', partition)
        assert len(dataset[0]) == 3
        assert dataset[0][0].shape == (1, penne.WINDOW_SIZE)
        assert dataset[0][1].shape == (1,)
        assert dataset[0][2].shape == (1, penne.AR_SIZE)

def test_ar_PTDB_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.ARDataset('PTDB', partition)
        assert len(dataset[0]) == 3
        assert dataset[0][0].shape == (1, penne.WINDOW_SIZE)
        assert dataset[0][1].shape == (1,)
        assert dataset[0][2].shape == (1, penne.AR_SIZE)

def test_ar_PTDB_voiced_getitem():
    for partition in ['test', 'train', 'valid']:
        dataset = data.ARDataset('PTDB', partition, True)
        assert len(dataset[0]) == 3
        assert dataset[0][0].shape == (1, penne.WINDOW_SIZE)
        assert dataset[0][1].shape == (1,)
        assert dataset[0][2].shape == (1, penne.AR_SIZE)

def test_ar_MDB_loader():
    batch_size = 3
    loader = data.ar_loader('MDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets, ar = next(it)
        assert features.shape == (batch_size, 1024)
        assert targets.shape == (batch_size,)
        assert ar.shape == (batch_size, penne.AR_SIZE)

def test_ar_PTDB_loader():
    batch_size = 3
    loader = data.ar_loader('PTDB', 'test', batch_size = batch_size, num_workers = 0)
    for i in tqdm.tqdm(range(10)):
        it = iter(loader)
        features, targets, ar = next(it)
        assert features.shape == (batch_size, 1024)
        assert targets.shape == (batch_size,)
        assert ar.shape == (batch_size, penne.AR_SIZE)