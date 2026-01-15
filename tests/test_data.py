from torch.utils.data import Dataset
from ml_ops_59.data import data_loader


def test_my_dataset():
    """Test the MyDataset class."""
    df = data_loader()
    assert isinstance(df, Dataset)


