from torch.utils.data import Dataset
from ml_ops_59.data import data_loader


def test_my_dataset():
    """Test the MyDataset class."""
    df = data_loader()
    print('this is just a test for the tests for Alberts test')
    #assert isinstance(df, Dataset)


