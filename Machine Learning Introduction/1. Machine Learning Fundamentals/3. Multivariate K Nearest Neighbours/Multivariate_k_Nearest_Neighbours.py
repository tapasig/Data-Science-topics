## 1. Recap ##

import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

''' Use the DataFrame.info() method to return the number of non-null values in each column.'''
dc_listings.info()

## 2. Removing features ##

#remove the columns
drop_columns = ['room_type', 'city','state','latitude','longitude','zipcode','host_response_rate','host_acceptance_rate','host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis=1)


#provide the sum all null numbers for each column
print(dc_listings.isnull().sum())


## 3. Handling missing values ##

#remove the columns
drop_columns = ['cleaning_fee','security_deposit']
dc_listings = dc_listings.drop(drop_columns, axis=1)

#remove rows with nan
dc_listings.dropna(axis=0,inplace=True)

#provide the sum all null numbers for each column
print(dc_listings.isnull().sum())

## 4. Normalize columns ##

# print(dc_listings.info())
# print(dc_listings.head())
# print(dc_listings['maximum_nights'].value_counts())

#Normalize all of the feature columns in dc_listings and assign the new Dataframe containing just the normalized feature columns to normalized_listings

print(dc_listings.std())
normalized_listings = (dc_listings - dc_listings.mean())/dc_listings.std()

#Add the price column from dc_listings to normalized_listings
normalized_listings['price'] = dc_listings['price']

normalized_listings.head(3)

## 5. Euclidean distance for multivariate case ##

from scipy.spatial import distance

# print(type(normalized_listings['accommodates']))
# print(normalized_listings.head())

print(normalized_listings['accommodates'].iloc[0])

first_listing = [normalized_listings['accommodates'].iloc[0], normalized_listings['bathrooms'].iloc[0]]

fifth_listing = [normalized_listings['accommodates'].iloc[4], normalized_listings['bathrooms'].iloc[4]]


first_fifth_distance = distance.euclidean(first_listing,fifth_listing )

print(first_fifth_distance)

## 7. Fitting a model and making predictions ##

from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]

'''Create an instance of the KNeighborsRegressor class with the following parameters:

n_neighbors: 5
algorithm: brute'''

knn = KNeighborsRegressor(algorithm='brute')

'''Use the fit method to specify the data we want the k-nearest neighbor model to use. Use the following parameters:
1. training data, feature columns: just the accommodates and bathrooms columns, in that order, from train_df.
2. training data, target column: the price column from train_df.'''

train_feature_columns = train_df[["accommodates","bathrooms"]]
train_target_column = train_df["price"]

knn.fit(train_feature_columns,train_target_column)

'''Call the predict method to make predictions on:

1. the accommodates and bathrooms columns from test_df
2. assign the resulting NumPy array of predicted price values to predictions.'''
predictions = knn.predict(test_df[["accommodates","bathrooms"]])

## 8. Calculating MSE using Scikit-Learn ##

from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])

'''Use the mean_squared_error function to calculate the MSE value for the predictions we made in the previous screen.
Assign the MSE value to two_features_mse.'''
from sklearn.metrics import mean_squared_error
#import math

two_features_mse = mean_squared_error(test_df['price'], predictions)

two_features_rmse = two_features_mse**(1/2)

print('two_features_mse=',two_features_mse," two_features_rmse=",two_features_rmse)

## 9. Using more features ##


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

#features and target from train dataset to fit the ML model
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']

train_features = train_df[features]
train_target = train_df['price']

#fit the model
knn.fit(train_features,train_target)

#use the ML model to predict the price using the features from test dataset
four_predictions = knn.predict(test_df[features])

#Evaluate the accuracy of predictions through error calculation
four_mse = mean_squared_error(test_df['price'],four_predictions)
four_rmse = four_mse**(1/2)

print("four_mse = ",four_mse,"four_rmse=",four_rmse)


## 10. Using all features ##

#train_df.head()

#Use all of the columns, except for the price column, to train a k-nearest neighbors model using the same parameters for the KNeighborsRegressor class 
features = ["accommodates","bedrooms","bathrooms","beds","minimum_nights","maximum_nights"
           ,"number_of_reviews"]

knn.fit(train_df[features],train_df["price"])

#predict the price using the model
all_features_predictions = knn.predict(test_df[features])

#Calculate the MSE and RMSE values and assign to all_features_mse and all_features_rmse accordingly.

all_features_mse = mean_squared_error(test_df['price'],all_features_predictions)

all_features_rmse = all_features_mse**(1/2)

print("all_features_mse = ",all_features_mse,"all_features_rmse =",all_features_rmse)
