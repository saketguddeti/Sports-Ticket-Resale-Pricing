## Model for ticket best value prediction

import pandas as pd
import numpy as np
import requests
import json
import os    
from datetime import datetime
os.chdir('/Users/saketguddeti/Desktop/git/Sports-Ticket-Resale-Pricing')

#########################
## StubHub API Parameters
auth_key="X9VaaNPfOR9BFq1rtyLujDIvmF0a"

headers = {
    "Authorization": "Bearer %s" % auth_key,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

host = "https://api.stubhub.com"

###################################################
## Extracting all the 'San Francisco Giants' events
endpoint = '/search/catalog/events/v3/?rows=500&q="San Francisco Giants" -PARKING&city=San Francisco'

url = "%s%s" % (host, endpoint)
r = requests.get(url, headers=headers)
json_obj = r.json()

data_giants = pd.DataFrame(json_obj['events'])[['id','eventDateUTC','eventDateLocal','name']]
data_giants['eventDateUTC'] = data_giants['eventDateUTC'].apply(
        lambda x: datetime.strptime(x[:-5], '%Y-%m-%dT%H:%M:%S'))
data_giants['eventDateLocal'] = data_giants['eventDateLocal'].apply(
        lambda x: datetime.strptime(x[:-5], '%Y-%m-%dT%H:%M:%S'))
data_giants['dateExtracted'] = datetime.today()
data_giants['DTG'] = (data_giants.eventDateUTC - data_giants.dateExtracted).apply(lambda x: x.days)

data_giants['Away_Team'] = data_giants['name'].str.split(' at ').apply(lambda x: x[0])
data_giants['Home_Team'] = 'San Francisco Giants'

fan_index = pd.read_excel('mlb_fans.xlsx')
data_giants = pd.merge(left=data_giants,right=fan_index,how='left',left_on='Away_Team',right_on='Team')
data_giants = pd.merge(left=data_giants,right=fan_index,how='left',left_on='Home_Team',right_on='Team')
data_giants['gameImp'] = data_giants['Fans_x'] + data_giants['Fans_y']

data_giants = data_giants.drop(['Fans_x','Fans_y','Team_x','Team_y'], axis = 1)
##################################################################
## Extracting ticket listings of all 'San Francisco Giants' events

# Listings sorted based on Best Value calculated by StubHub

listing_giants = pd.DataFrame()
for n,i in enumerate(list(data_giants['id'])):
    print("Appended "+str(n)+" out of "+str(len(data_giants['id'])))
    for j in [0+250*k for k in range(12)]:
        try:
            print('Appended '+str(j))
            endpoint = "/search/inventory/v2/?eventId=%i&start=%i&rows=250&sort=value+desc" % (i, j)
            url = "%s%s" % (host, endpoint)
            r = requests.get(url, headers=headers)
            json_obj = r.json()
            listing_data = pd.DataFrame(json_obj['listing'])
            listing_data['eventID'] = i
            listing_giants = pd.concat([listing_giants,listing_data], axis = 0)
        except:
            print("Skipping")
            break

listing_giants['Listing_Price'] = listing_giants['listingPrice'].apply(lambda x: x['amount'])
listing_giants = listing_giants.drop(['businessGuid','deliveryMethodList','currentPrice','deliveryTypeList',
                      'dirtyTicketInd','faceValue','isGA','score','sellerOwnInd',
                      'sellerSectionName','listingPrice'], axis = 1)
listing_giants['value'] = listing_giants.groupby(['eventID']).cumcount() + 1
listing_giants['value'] = listing_giants.groupby(['eventID']).value.transform(lambda x: 1 - x/x.max())

listing_giants2 = pd.merge(left = listing_giants, right = data_giants[['id','DTG','eventDateLocal','gameImp']],
                          how = 'left', left_on='eventID', right_on='id')

listing_giants2['Day_Of_Week'] = listing_giants2['eventDateLocal'].apply(lambda x: x.weekday()).map(
        {0:'Weekday',
         1:'Weekday',
         2:'Weekday',
         3:'Weekday',
         4:'Weekday',
         5:'Weekend',
         6:'Weekend'})
    
listing_giants2['Time_Of_Day'] = listing_giants2['eventDateLocal'].apply(lambda x: x.hour)
listing_giants2['Time_Of_Day'] = np.where(listing_giants2['Time_Of_Day'].isin([12,13]), 'Lunch-Time', 'Prime-Time')


# Zone Importance
zone_imp = listing_giants2.groupby(['zoneId'])['Listing_Price'].median().reset_index()
zone_imp.columns = ['zoneId','ZoneImp']
listing_giants2 = pd.merge(left=listing_giants2,right=zone_imp,how='left',on='zoneId')
listing_giants2 = listing_giants2.loc[pd.notnull(listing_giants2['ZoneImp'])]

# Section Importance
sec_imp = listing_giants2.groupby(['sectionId'])['Listing_Price'].median().reset_index()
sec_imp.columns = ['sectionId','secImp']
listing_giants2 = pd.merge(left=listing_giants2,right=sec_imp,how='left',on='sectionId')
listing_giants2 = listing_giants2.loc[pd.notnull(listing_giants2['secImp'])]

# Available Listings
avail_listings = listing_giants2.groupby(['eventID','zoneId']).size().reset_index(name = 'Total_Listings')
listing_giants2 = pd.merge(left=listing_giants2,right=avail_listings,how='left',on=['eventID','zoneId'])

total_cap = listing_giants2[['zoneId','row','seatNumbers','quantity']].drop_duplicates()
total_cap = total_cap.groupby('zoneId').quantity.sum().reset_index(name = 'Total_Capacity')
listing_giants2 = pd.merge(left=listing_giants2,right=total_cap,how='left',on='zoneId')

listing_giants2['availIndex'] = 100 * listing_giants2['Total_Listings'] / listing_giants2['Total_Capacity']

listing_giants2.to_csv('listing_giants.csv', index = False)

X = listing_giants2[['Listing_Price','DTG','Day_Of_Week','Time_Of_Day','ZoneImp',
                     'gameImp','secImp']]
y = listing_giants2.value

# Encoding Time_Of_Day variable
DUMMY = pd.get_dummies(X['Time_Of_Day'])
X = pd.concat([X, DUMMY], axis = 1)

# Encoding Day_Of_Week variable
DUMMY = pd.get_dummies(X['Day_Of_Week'])
X = pd.concat([X, DUMMY], axis = 1)

X = X.drop(['Time_Of_Day','Day_Of_Week'], axis = 1)

# Splitting the Data in train/test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)


## Scaling the data
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X_train['ZoneImp'].values.reshape(-1,1))
#X_train.loc[:,'ZoneImp'] = scaler.transform(X_train['ZoneImp'].values.reshape(-1,1))
#X_test.loc[:,'ZoneImp'] = scaler.transform(X_test['ZoneImp'].values.reshape(-1,1))



# Building Model (Decision Tree Regressor)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
min_sam_splt = np.arange(2,100,10)
min_sam_leaf = np.arange(1,50,10)
depth = np.arange(2,30,5)
param_grid = {'min_samples_split': min_sam_splt, 
              'min_samples_leaf' : min_sam_leaf,
              'max_depth':depth
              }
regressor = DecisionTreeRegressor()
regressor_cv = GridSearchCV(regressor, param_grid, cv=5)
regressor_cv.fit(X_train, y_train)
print("Tuned Decision Tree Regressor Parameters: {}".format(regressor_cv.best_params_)) 
print("Best score is {}".format(regressor_cv.best_score_))

regression_tree = DecisionTreeRegressor(max_depth=20,min_samples_split=32,min_samples_leaf=2,random_state=2)
regression_tree.fit(X_train,y_train)

from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error
y_pred = regression_tree.predict(X_test)
y_pred_train = regression_tree.predict(X_train)

print("R-Squared value for Train: {}, Test: {}".format(r2_score(y_train, y_pred_train), 
      r2_score(y_test, y_pred)))

# Feature Importance
x = zip(X_train.columns, regression_tree.feature_importances_)
sorted(x, key = lambda x: x[1], reverse = True)

# Prediction
x = pd.DataFrame({'Listing_Price':[53.25],
                  'DTG':[32],
                  'ZoneImp':[95],
                  'gameImp':[531.84],
                  'secImp':[111.83],
                  'Lunch-Time':[0],
                  'Prime-Time':[1],
                  'Weekday':[1],
                  'Weekend':[0]})

xx = pd.DataFrame({'Listing_Price':[80.0],
                   'DTG':[137],
                   'ZoneImp':[95],
                   'gameImp':[469.76],
                   'secImp':[90.53],
                   'Lunch-Time':[0],
                   'Prime-Time':[1],
                   'Weekday':[0],
                   'Weekend':[1]})

X_train.iloc[0,:] = [140,32,95,531.84,99.5,0,1,1,0]
x = X_train.head(1)
regression_tree.predict(X_train)



x = listing_giants2.loc[(listing_giants2['eventID'] == 103233999) &
                        (listing_giants2['sectionName'] == 'Club Level Infield 211')]

X_train.loc[X_train.index == 69056]
y_pred_train.loc[y_pred_train.index == 69056]
