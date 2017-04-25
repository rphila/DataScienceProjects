# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:37:43 2017

@author: Rita Philavanh
"""
import os
os.chdir('C:/Users/main/Documents/Data Science Program/Data Science Challenge')
### SETUP: Download the Adventure Works data files into working dir
import BikeBuyerUtil as bb
import numpy as np


DoClassification = True #False (if Regression)

#############
# LOAD DATA
#############
import pandas as pd

print('Loading Customer and Sales dataset...')
df_Cus = pd.read_csv('AWCustomers.csv') 
df_Sales = pd.read_csv('AWSales.csv') 
print(len(df_Cus), ' customer records')
print(len(df_Cus['CustomerID'].unique()), ' unique customer records')
print(len(df_Sales), ' customer sales records')

##############
## CLEAN DATA
##############
print('Removing duplicate CustomerIDs (while keeping latest updated records)...')
dup_ID = df_Cus.CustomerID[df_Cus.CustomerID.duplicated()]
df_Cus[df_Cus.CustomerID.isin(dup_ID)].sort("CustomerID")
df_Cus = df_Cus[-(df_Cus.CustomerID.isin(dup_ID) & (df_Cus.LastUpdated == '2017-03-06'))]

print('Examining if any fields contain problematic null values....')
print(df_Cus.columns[pd.isnull(df_Cus).sum() > 0] )
print(df_Sales.columns[pd.isnull(df_Sales).sum() > 0] )
print('No handling of null values needed. Nulls were only observed in non problematic fields: Title, MiddleName, Suffix, AddressLine2')


##############
## TRANSFORM DATA
##############
print('Examining datatypes for each fields....')
print(df_Cus.dtypes)
print(df_Sales)

print('Converting dates to datetime data type....')
df_Cus.LastUpdated = pd.to_datetime(df_Cus.LastUpdated, errors='coerce')
df_Cus.BirthDate = pd.to_datetime(df_Cus.BirthDate, errors='coerce')
print('Adding calculated Age column type....')
df_Cus['Age'] = (df_Cus.LastUpdated - df_Cus.BirthDate).astype('<m8[Y]')


print('Converting ordered categorical fields to numeric category so they can be more easily handled during analysis....')
print(df_Cus.Education.unique())
ordered_Education = ['Partial High School','High School','Partial College', 
        'Bachelors', 'Graduate Degree']

df_Cus.Education = df_Cus.Education.astype("category",
  ordered=True,
  categories=ordered_Education
  ).cat.codes

ordered_Occupation = ['Manual', 'Skilled Manual','Clerical',  'Management', 'Professional']

df_Cus.Occupation = df_Cus.Occupation.astype("category",
  ordered=True,
  categories=ordered_Occupation
  ).cat.codes
                                           
print('Exploding nominal categories into boolean features to be handled duyring analysis (since there is no inherent order to track when making them categories)...')
df_Cus = pd.concat([df_Cus[['Gender', 'MaritalStatus']],pd.get_dummies(df_Cus,columns=['Gender','MaritalStatus'])],axis=1)

print('Computing additional boolean fields that may potentially have relationships...')
df_Cus['HasChildrenAtHome'] = 0 
df_Cus['HasCars'] = 0
df_Cus.ix[df_Cus.NumberChildrenAtHome > 0, 'HasChildrenAtHome'] = 1
df_Cus.ix[df_Cus.NumberCarsOwned > 1, 'HasCars'] = 1
          
print('Joining Customer and Sales data on CustomerID....')
df_join = pd.merge(df_Cus, df_Sales, on='CustomerID', how='left')
print(df_join.dtypes)
print(len(df_join))


##############
## PREPROCESS DATA
# Note: Different models were previously tested with parameter sweeping performed to identify optimal model and parameter settings
##############

#Assign feature set
print('Droping unessential features for modeling (e.g., ID and non-category text columns)....')
X = df_join.drop(labels=['CustomerID'
                         ,'Title'
                         ,'FirstName'
                         ,'MiddleName'
                         ,'LastName'
                         ,'Suffix'
                         ,'AddressLine1'
                         ,'AddressLine2'
                         ,'City'
                         ,'StateProvinceName'
                         ,'CountryRegionName'
                         ,'PostalCode'
                         ,'PhoneNumber'
                         ,'BirthDate'
                         ,'LastUpdated'
                         ,'Gender'
                         ,'MaritalStatus'
                         #'Occupation'
                         ,'BikeBuyer'
                         ,'AvgMonthSpend'
                         #,'NumberCarsOwned'
                         #,'NumberChildrenAtHome'
                         #,'TotalChildren'
                         #,'Education'
                         ], axis=1)

#Assign label

if DoClassification:
    y = df_join.BikeBuyer  
else:
    y = df_join.AvgMonthSpend 
        
    
print('Spliting into train/test set...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
print(len(X_train))
print(len(X_test))


##############
## MACHINE LERANING MODEL TRAINING AND TESTING
# Note: Different models were previously tested with parameter sweeping performed to identify optimal model and parameter settings
##############

if DoClassification:
    print("Performing Regression Modeling...")

    from sklearn import ensemble
    from sklearn.metrics import mean_squared_error

    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': .02, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)

    print('Calculating  reggression modeling results...")
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    print("RMSE: ",np.sqrt(mse))

    print('Plotting True vs Predicted values...")
    import numpy as np
    import matplotlib.pylab as plt
    plt.scatter(np.log(y_test), np.log(clf.predict(X_test)))
    plt.xlabel('True AvgMonthSpend')
    plt.ylabel('Predicted AvgMonthSpend')
    
    #predictions = clf.predict(data_test)
    #pd.DataFrame(predictions).to_csv('AWS_Reg_resultRandomForestReg.csv')

else:
    print('Performing Classification Modeling...")

    from sklearn import tree
    model = tree.DecisionTreeClassifier(max_depth=6, criterion="entropy")
    model.fit(X_train, y_train)

    print('Calculating calssification modeling results...")
    from sklearn.metrics import confusion_matrix, precision_score,recall_score, accuracy_score, classification_report, f1_score
    #print(model.score(X_test, y_test))
    prediction = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, prediction))
    print('F1 score:', f1_score(y_test, prediction))
    print('Recall:', recall_score(y_test, prediction))
    print('Precision:', precision_score(y_test, prediction))
    print('\n clasification report:\n', classification_report(y_test,prediction))
    print('\n confussion matrix:\n',confusion_matrix(y_test, prediction))
    
    print('Plotting ROC curve...")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, prediction)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve' )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()