import unittest
import os
import pandas as pd
from tqdm import tqdm


class TestDataPreprocess(unittest.TestCase):
    def test_data_preprocess(self):
        # read reference csv
        from data_preprocess import parent_dir_name, trajectories_out_file_name
        # check csv without_fill first
        assert os.path.exists(os.path.join(parent_dir_name, trajectories_out_file_name + "_without_fill.csv"))
        my_csv = pd.read_csv(os.path.join(parent_dir_name, trajectories_out_file_name + "_without_fill.csv"))
        # select column vehicle_id, time, x, y from my csv
        my_csv = my_csv[['vehicle_id', 'time', 'x', 'y']]
        reference_csv = self.load_reference(parent_dir_name, "reference_without_fill.csv")
        self.assertTrue(my_csv.equals(reference_csv))
        assert os.path.exists(os.path.join(parent_dir_name, trajectories_out_file_name + ".csv"))
        my_csv = pd.read_csv(os.path.join(parent_dir_name, trajectories_out_file_name + ".csv"))
        # select column vehicle_id, time, x, y from my csv
        my_csv = my_csv[['vehicle_id', 'time', 'x', 'y']]
        reference_csv = self.load_reference(parent_dir_name, "reference.csv")
        # compare two csv line by line
        for i in tqdm(range(len(my_csv))):
            my_row = my_csv.iloc[i]
            reference_row = reference_csv.iloc[i]
            msg = f"my_csv {my_row} is not equal to reference_csv {reference_row} at {i}"
            self.assertTrue(my_row.equals(reference_row), msg)
        # select row with time=0 and vehicle_id=0

    def load_reference(self, parent_dir_name, filename):
        assert os.path.exists(os.path.join(parent_dir_name, filename))
        reference_csv = pd.read_csv(str(os.path.join(parent_dir_name, filename)))
        # rename the columns in reference_csv (longitude, latitude) to (x, y)
        reference_csv = reference_csv.rename(columns={'longitude': 'x', 'latitude': 'y'})
        reference_csv = reference_csv[['vehicle_id', 'time', 'x', 'y']]
        # change reference csv vehicle_id column to dtype int
        reference_csv['vehicle_id'] = reference_csv['vehicle_id'].astype(int)
        return reference_csv