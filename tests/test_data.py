import numpy as np
import penne
import penne.data as data
import penne.convert as convert
import tqdm


###############################################################################
# Test data.py
###############################################################################


# def test_getitem():
#     mdb_dataset = data.Dataset('MDB', 'valid')
#     ptdb_dataset = data.Dataset('PTDB', 'valid')
#     assert len(mdb_dataset[0]) == 2
#     assert convert.samples_to_frames(mdb_dataset[0][0].shape[1]-penne.WINDOW_SIZE) == mdb_dataset[0][1].shape[1]

#     assert len(ptdb_dataset[0]) == 2
#     assert convert.samples_to_frames(ptdb_dataset[0][0].shape[1]-penne.WINDOW_SIZE) == ptdb_dataset[0][1].shape[1]

# def test_getitem_slice():
#     mdb_dataset = data.Dataset('MDB', 'valid', random_slice=True)
#     ptdb_dataset = data.Dataset('PTDB', 'valid', random_slice=True)
#     assert len(mdb_dataset[0]) == 2
#     assert convert.samples_to_frames(mdb_dataset[0][0].shape[1]-penne.WINDOW_SIZE) == mdb_dataset[0][1].shape[1]

#     assert len(ptdb_dataset[0]) == 2
#     assert convert.samples_to_frames(ptdb_dataset[0][0].shape[1]-penne.WINDOW_SIZE) == ptdb_dataset[0][1].shape[1]

def test_MDB_loader():
    batch_size = 3
    loader = data.loader('MDB', 'valid', batch_size = batch_size, num_workers = 1)
    it = iter(loader)
    features, targets = next(it)
    assert features.shape == (batch_size, 1024, convert.seconds_to_frames(1))
    assert targets.shape == (batch_size, convert.seconds_to_frames(1))

def test_PTDB_loader():
    batch_size = 3
    loader = data.loader('PTDB', 'valid', batch_size = batch_size, num_workers = 1)
    for i in tqdm.tqdm(range(50)):
        it = iter(loader)
        features, targets = next(it)
        assert features.shape == (batch_size*convert.seconds_to_frames(1), 1024)
        assert targets.shape == (batch_size*convert.seconds_to_frames(1),)