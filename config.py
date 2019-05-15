import os

root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'data')

raw_data_dir = os.path.join(data_dir, 'raw')

location_data_dir = os.path.join(raw_data_dir, 'FSDT_FinAccessMapping')
