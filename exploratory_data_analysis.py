import import_data
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from import_data import load_housing_data
from setup import save_fig

def eda():
	# exploratory data analysis
	housing = load_housing_data()
	print (housing.head())
	print (housing.info())
	print (housing.describe())
	print (housing['ocean_proximity'].value_counts())

	# graphical data analysis

	housing.hist(bins=50, figsize=(20,15))
	save_fig("attribute_histogram_plots")
	plt.show()


if __name__ == '__main__':
	eda()