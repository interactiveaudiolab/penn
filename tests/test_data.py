import numpy as np
import penne.data as data


###############################################################################
# Test data.py
###############################################################################


# def test_getitem():
#     mdb_dataset = data.Dataset('MDB', 'test')
#     ptdb_dataset = data.Dataset('PTDB', 'test')
#     import pdb; pdb.set_trace()
#     assert len(mdb_dataset[0]) == 2
#     assert len(ptdb_dataset[0]) == 2

def test_loader():
    batch_size = 2

    def test_loader_helper(dataset):
        batch_size = 2
        loader = data.loader(dataset, 'test', batch_size = batch_size, num_workers = 1)
        it = iter(loader)
        features, targets = next(it)
        assert features.shape == (batch_size, 1024, 100)
        assert targets.shape == (batch_size, 100)

    test_loader_helper('MDB')
    test_loader_helper('PTDB')