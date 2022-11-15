
from .base import AbstractDataset

import pandas as pd

class RetailDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'retail'

    def load_df(self):
        folder_path = self._get_rawdata_root_path()
        file_path = folder_path.joinpath('retail.txt')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'behavior', 'timestamp']
        return df