import numpy as np 
from import_data import load_housing_data
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(42)
housing = load_housing_data()

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# import hashlib
def test_set_check(identifier, test_ratio, hash):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# This version supports both Python 2 and Python 3, instead of just Python 3.
def illustration_test_train_split():
	train_set, test_set = split_train_test(housing, 0.2)
	print(len(train_set), "train +", len(test_set), "test")

	housing_with_id = housing.reset_index()   # adds an `index` column
	train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

	housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
	train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

	print (test_set.head())

# use scikit learn train test split

#from sklearn.model_selection import train_test_split

def scikit_train_test_split():
	train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
	print (test_set.head())

	return train_set, test_set

# train_test_split is useful when we have a huge data
# but for small data set, it may end up causing sampling bias
# hence we wil use stratified sampling

def stratified_train_test_split():

	housing["median_income"].hist()
	plt.show()

	# most of income are in range 2 to 5. It won't spread across equivalent
	# strata. 
	# Divide by 1.5 to limit the number of income categories
	housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
	# Label those above 5 as 5
	housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
	print (housing["income_cat"].value_counts())
	housing["income_cat"].hist()
	plt.show()

	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(housing, housing["income_cat"]):
	    strat_train_set = housing.loc[train_index]
	    strat_test_set = housing.loc[test_index]

	print (housing["income_cat"].value_counts() / len(housing))

	return strat_train_set,strat_test_set


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

def train_test_split_comparsion(test_set,strat_test_set):

	compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
	}).sort_index()
	compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
	compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

	print (compare_props)


if __name__ == '__main__':

	illustration_test_train_split()
	#train_set, test_set= scikit_train_test_split()
	strat_train_set,strat_test_set= stratified_train_test_split()
	train_set, test_set= scikit_train_test_split()
	# print (strat_test_set.columns)
	# print(test_set["income_cat"],strat_test_set["income_cat"])

	train_test_split_comparsion(test_set,strat_test_set)

	
	





