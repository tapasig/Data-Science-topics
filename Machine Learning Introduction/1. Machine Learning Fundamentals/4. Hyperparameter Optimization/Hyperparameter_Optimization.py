## 1. Recap ##

#read the csv files
import pandas as pd
train_df = pd.read_csv("dc_airbnb_train.csv")
test_df = pd.read_csv("dc_airbnb_test.csv")

## 2. Hyperparameter optimization ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

#Create a list containing the integer values 1, 2, 3, 4, and 5
hyper_params = [1, 2, 3, 4, 5]


'''Use a for loop to iterate over hyper_params and in each iteration:
1. instantiate a knn model
2. fit the model with training features and target
3. predict the target from the test data
4. evaluate the performance and fill the mse_values list'''

#Create an empty list and assign to mse_values
mse_values = []

features = ["accommodates","bedrooms","bathrooms","number_of_reviews"]

for k in hyper_params:
    
    #instantiate the model
    knn = KNeighborsRegressor(n_neighbors=k, algorithm = "brute")
    
    #fit the model with features with a target as price
    knn.fit(train_df[features],train_df["price"])
    
    #predict the price using test data
    predictions = knn.predict(test_df[features])
    
    #evaluate the prediction
    mse = mean_squared_error(test_df["price"],predictions)
    
    #fill the list
    mse_values.append(mse)

print(mse_values)

## 3. Expanding grid search ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

#Create a list containing the integer values 1 to 20
hyper_params = list(range(1,21))


'''Use a for loop to iterate over hyper_params and in each iteration:
1. instantiate a knn model
2. fit the model with training features and target
3. predict the target from the test data
4. evaluate the performance and fill the mse_values list'''

#Create an empty list and assign to mse_values
mse_values = []

features = ["accommodates","bedrooms","bathrooms","number_of_reviews"]

for k in hyper_params:
    
    #instantiate the model
    knn = KNeighborsRegressor(n_neighbors=k, algorithm = "brute")
    
    #fit the model with features with a target as price
    knn.fit(train_df[features],train_df["price"])
    
    #predict the price using test data
    predictions = knn.predict(test_df[features])
    
    #evaluate the prediction
    mse = mean_squared_error(test_df["price"],predictions)
    
    #fill the list
    mse_values.append(mse)

print(mse_values)

## 4. Visualizing hyperparameter values ##

import matplotlib.pyplot as plt

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [x for x in range(1, 21)]
mse_values = list()

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
    
#confirm this behavior of increasing k visually using a scatter plot  
plt.scatter(hyper_params,mse_values)
plt.xlabel("K")
plt.show()

## 5. Varying Hyperparameters ##

hyper_params = [x for x in range(1,21)]
mse_values = list()


features = ['accommodates', 'bedrooms', 'bathrooms','beds',
            'minimum_nights', 'maximum_nights', 'number_of_reviews']

hyper_params = [x for x in range(1, 21)]
mse_values = list()

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
    
#confirm this behavior of increasing k visually using a scatter plot  
plt.scatter(hyper_params,mse_values)
plt.xlabel("K")
plt.ylabel("MSE")
plt.show()

## 6. Practice the workflow ##

two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
# Append the second model's MSE values to this list.
three_mse_values = list()
two_hyp_mse = dict()
three_hyp_mse = dict()

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    mse = mean_squared_error(test_df['price'], predictions)
    two_mse_values.append(mse)
    
    
for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])
    mse = mean_squared_error(test_df['price'], predictions)
    three_mse_values.append(mse)
    
#find the min mse values
min_two_mse_values = min(two_mse_values)
min_three_mse_values = min(three_mse_values)

#print(min_two_mse_values, min_three_mse_values)


for k, value in enumerate(two_mse_values):
    #print(k, value)
    if value == min_two_mse_values:
        two_hyp_mse[k+1] = value
        
for k, value in enumerate(three_mse_values):
    if value == min_three_mse_values:
        three_hyp_mse[k+1] = value
        
print("two_hyp_mse=",two_hyp_mse," three_hyp_mse= ",three_hyp_mse)
