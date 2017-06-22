from create_test_set import stratified_train_test_split,drop_income_cate
from import_data import load_housing_data
import matplotlib.pyplot as plt
from setup import save_fig
import matplotlib.image as mpimg
import numpy as np 
from pandas.plotting import scatter_matrix

PROJECT_ROOT_DIR = "."


housing = load_housing_data()
strat_train_set,strat_test_set = stratified_train_test_split(housing)
strat_train_set,strat_test_set= drop_income_cate(strat_train_set,strat_test_set)

housing = strat_train_set.copy()
print (housing.info())

housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
# save_fig("bad_visualization_plot")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
# save_fig("better_visualization_plot")


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()
# save_fig("housing_prices_scatterplot")


california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
# save_fig("california_housing_prices_plot")
plt.show()

corr_matrix = housing.corr()

print (corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
# save_fig("scatter_matrix_plot")


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


	