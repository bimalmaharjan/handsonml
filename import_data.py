
import pandas as pd
import os
from 1_download_data import HOUSING_PATH 

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == '__main__':
	housing = load_housing_data()
	print (housing.head())
	print (housing.info)

