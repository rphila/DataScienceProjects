# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:37:43 2017

@author: Rita Philavanh
"""

###!!!
### SETUP: Download the Adventure Works data files into working dir
#import os
#os.chdir('C:/Users/main/Documents/Data Science Program/Data Science Challenge')

import BikeBuyerUtil as bb
import numpy as np


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
print('Examining any duplicate CustomerIDs...')
print(len(df_Cus.CustomerID[df_Cus.CustomerID.duplicated()]), ' duplicate CustomerID in Customer dataset')
print(len(df_Sales.CustomerID[df_Sales.CustomerID.duplicated()]), ' duplicate CustomerID in Sales dataset')

print('Removing duplicate CustomerIDs (while keeping latest updated records)...')
#df_Cus = df_Cus.drop_duplicates(subset=['CustomerID'])
dup_ID = df_Cus.CustomerID[df_Cus.CustomerID.duplicated()]
df_Cus[df_Cus.CustomerID.isin(dup_ID)].sort("CustomerID")
df_Cus = df_Cus[-(df_Cus.CustomerID.isin(dup_ID) & (df_Cus.LastUpdated == '2017-03-06'))]
print(len(df_Cus), ' customer records')

print('Examining if any fields contain problematic null values....')
print(df_Cus.columns[pd.isnull(df_Cus).sum() > 0] )
print(df_Sales.columns[pd.isnull(df_Sales).sum() > 0] )
print('No handling of null values needed. Nulls were only observed in non problematic fields: Title, MiddleName, Suffix, AddressLine2')
#print(df_Cus[pd.isnull(df_Cus).any(axis=1)])
#print(df_Sales[pd.isnull(df_Sales).any(axis=1)])


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

print('Exploding nominal categories into boolean features to be handled duyring analysis (since there is no inherent order to track when making them categories)...')
df_Cus = pd.concat([df_Cus[['Gender','Occupation','MaritalStatus']],pd.get_dummies(df_Cus,columns=['Gender','Occupation','MaritalStatus'])],axis=1)

print('Computing additional boolean fields that may potentially have relationships...')
df_Cus['HasChildrenAtHome'] = 0 
df_Cus['HasCar'] = 0
df_Cus.ix[df_Cus.NumberChildrenAtHome > 0, 'HasChildrenAtHome'] = 1
df_Cus.ix[df_Cus.NumberCarsOwned > 1, 'HasCar'] = 1
          
print('Joining Customer and Sales data on CustomerID....')
df_join = pd.merge(df_Cus, df_Sales, on='CustomerID', how='left')
print(df_join.dtypes)
print(len(df_join))

##############
## EXPLORE DATA THROUGH STATISTICS AND VISUALIZATION
##############
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 

import seaborn as sns
num_cols = ['Age','Education',  'HasChildrenAtHome', 'HasCar', 'BikeBuyer', 'AvgMonthSpend'] 
sns.pairplot(df_join[num_cols], size=2)

print('Plot corrleation matrix....')
featureSet1 = ['Age','Education', 'HasChildrenAtHome', 'HasCar', 'BikeBuyer', 'AvgMonthSpend']
bb.correlation_matrix(df_join, featureSet1, 'Corr1')

featureSet2 = ['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'BikeBuyer', 'AvgMonthSpend']
bb.correlation_matrix(featureSet2, 'Corr2')

featureSet3 = ['Gender_F', 'Gender_M', 'MaritalStatus_M', 'MaritalStatus_S', 'BikeBuyer', 'AvgMonthSpend']
bb.correlation_matrix(featureSet3, 'Corr3')

featureSet4 = ['Occupation_Clerical', 'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional', 'Occupation_Skilled Manual', 'BikeBuyer', 'AvgMonthSpend']
bb.correlation_matrix(featureSet4, 'Corr4')


print('Relationship between #Children and #Car with BikeBuyer')
fig = plt.figure()
fig.suptitle('Frequency', fontsize=20)
ax1 = fig.add_subplot(221) #row,#col,fig#
ax1.hist(df_join[(df_join.BikeBuyer == 0)].NumberChildrenAtHome, color='red')
ax1.set_ylabel('Non Bike Buyers')
#ax1.set_xlabel('# Children At Home')

ax3 = fig.add_subplot(222) #row,#col,fig#
ax3.hist(df_join[(df_join.BikeBuyer == 0)].TotalChildren, color='blue')
#ax3.set_xlabel('Total Children')

ax2 = fig.add_subplot(223) #row,#col,fig#
ax2.hist(df_join[(df_join.BikeBuyer == 1)].NumberChildrenAtHome, color='red')
ax2.set_ylabel('Bike Buyers')
ax2.set_xlabel('# Children At Home')

ax4 = fig.add_subplot(224) #row,#col,fig#
ax4.hist(df_join[(df_join.BikeBuyer == 1)].TotalChildren, color='blue')
ax4.set_xlabel('Total Children')
plt.show()

ax5 = fig.add_subplot(325) #row,#col,fig#
ax5.hist(df_join[(df_join.BikeBuyer == 0)].NumberCarsOwned, color='green')
ax5.set_ylabel('# Cars')

ax6 = fig.add_subplot(326) #row,#col,fig#
ax6.hist(df_join[(df_join.BikeBuyer == 1)].NumberCarsOwned, color='green')

plt.show()


print('Relationship between #Children and #Car and AvgMonthSpend')
print(df_join['AvgMonthSpend'].groupby(df_join.NumberCarsOwned).median().sort_values())
print(df_join['AvgMonthSpend'].groupby(df_join.NumberChildrenAtHome).median().sort_values())
print(df_join['AvgMonthSpend'].groupby(df_join.NumberCarsOwned).min().sort_values())
print(df_join['AvgMonthSpend'].groupby(df_join.NumberChildrenAtHome).min().sort_values())


print('Relationship between Gender, Age and AvgMonthSpend')
fig = plt.figure()
fig.suptitle('X vs AvgMonthSpend', fontsize=20)
ax1 = fig.add_subplot(211) #row,#col,fig#
ax1.scatter(df_join.NumberCarsOwned, df_join.AvgMonthSpend, color='red')
ax1.set_xlabel('X = NumberCarsOwned')
#ax1.set_ylabel('AvgMonthSpend')

ax2 = fig.add_subplot(212)
ax2.scatter(df_join.NumberChildrenAtHome, df_join.AvgMonthSpend, color='red')
ax2.set_xlabel('X = NumberChildrenAtHome')
plt.show()


fig = plt.figure()
fig.suptitle('Frequency of Bike Buyers', fontsize=20)
ax1 = fig.add_subplot(121) #row,#col,fig#
ax1.hist(df_join[(df_join.BikeBuyer == 0)].NumberChildrenAtHome, color='red')
ax1.set_title('Non Bike Buyers')
ax1.set_xlabel('# Children At Home')

ax2 = fig.add_subplot(122) #row,#col,fig#
ax2.hist(df_join[(df_join.BikeBuyer == 1)].NumberChildrenAtHome, color='red')
ax2.set_title('Bike Buyers')
ax2.set_xlabel('# Children At Home')
plt.show()

    
print('Summary statistics of AvgMonthSpend')
print(df_join.AvgMonthSpend.describe())

print('Visual distribution of AvgMonthSpend and BikeBuyers')
df_join[['AvgMonthSpend']].boxplot()
df_join.BikeBuyer.plot.hist()

print('Relationship between Occupation, Income and AvgMonthSpend')
print(df_join['YearlyIncome'].groupby(df_join.Education).median().sort_values())
print(df_join['AvgMonthSpend'].groupby(df_join.Education).median().sort_values())
# High income = High AvgMonthSpend

df_join[['AvgMonthSpend','Occupation']].boxplot(by='Occupation')
df_join[['AvgMonthSpend','YearlyIncome']].boxplot(by='YearlyIncome')


print('Relationship between Gender, Age and AvgMonthSpend')
fig = plt.figure()
fig.suptitle('Age vs AvgMonthSpend', fontsize=20)
ax1 = fig.add_subplot(211) #row,#col,fig#
ax1.scatter(df_join[(df_join.Gender=='F')].Age, df_join[(df_join.Gender=='F')].AvgMonthSpend, color='red')
ax1.set_title('Female')

ax2 = fig.add_subplot(212)
ax2.scatter(df_join[(df_join.Gender=='M')].Age, df_join[(df_join.Gender=='M')].AvgMonthSpend, color='blue')
ax2.set_title('Male')
plt.show()

# Save the figure
fig.savefig('AgeVsAvgMonthSpendByGender.png', bbox_inches='tight')
          
  
#3D Scatter
from mpl_toolkits.mplot3d import Axes3D

fig3D = plt.figure()
ax = fi3Dg.add_subplot(111, projection='3d')
ax.set_xlabel('Age')
ax.set_ylabel('AvgMonthSpend')
ax.set_zlabel('Gender')

ax.scatter(df_join.Age, df_join.AvgMonthSpend, (df_join.Gender == 'F').astype(int), c=(df_join.Gender == 'F').astype(int), marker='.', alpha=.5)
plt.show()

#df_join['AvgMonthSpend'].groupby(df_join.MaritalStatus).hist()


df_join[['HomeOwnerFlag','AvgMonthSpend']].groupby('HomeOwnerFlag').describe()
df_join[['Gender','AvgMonthSpend']].groupby('Gender').describe()
df_join[['MaritalStatus','AvgMonthSpend']].groupby('MaritalStatus').describe()


grouped = df_join[['Gender','AvgMonthSpend']].groupby('Gender')
grouped.boxplot()

grouped = df_join[['MaritalStatus','AvgMonthSpend']].groupby('MaritalStatus')
grouped.boxplot()
grouped = df_join[['Occupation','AvgMonthSpend']].groupby('Occupation')
grouped.boxplot()
grouped = df_join[['Education','AvgMonthSpend']].groupby('Education')
grouped.boxplot()
###
grouped = df_join[['NumberCarsOwned','AvgMonthSpend']].groupby('NumberCarsOwned')
grouped.boxplot()
grouped = df_join[['NumberChildrenAtHome','AvgMonthSpend']].groupby('NumberChildrenAtHome')
grouped.boxplot()
grouped = df_join[['TotalChildren','AvgMonthSpend']].groupby('TotalChildren')
grouped.boxplot()

print('Relationship of AvgMonthSpend based on demographics')

print(df_join['AvgMonthSpend'].groupby(df_join.MaritalStatus).count())
print(df_join['AvgMonthSpend'].groupby(df_join.MaritalStatus).median())
print(df_join['AvgMonthSpend'].groupby(df_join.HasCar).median())
print(df_join['AvgMonthSpend'].groupby(df_join.Gender).median())
print(df_join['AvgMonthSpend'].groupby(df_join.Gender).min())
print(df_join['AvgMonthSpend'].groupby(df_join.Gender).max())
print(df_join['AvgMonthSpend'].groupby(df_join.HasChildrenAtHome).median())

print(df_join['AvgMonthSpend'].groupby(df_join.StateProvinceName).median())
print(df_join['AvgMonthSpend'].groupby(df_join.StateProvinceName).count())

print('Relationship of BikeBuyer based on demographics')
print(df_join['YearlyIncome'].groupby(df_join.BikeBuyer).median())
print(df_join['NumberCarsOwned'].groupby(df_join.BikeBuyer).median())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.Occupation).BikeBuyer.count())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.Gender).BikeBuyer.count())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.MaritalStatus).BikeBuyer.count())

print(df_join[df_join.BikeBuyer == 1].groupby(df_join.Education).BikeBuyer.count().sort_values())

print('Relationship of BikeBuyer based on demographics')
print(df_join['YearlyIncome'].groupby(df_join.BikeBuyer).median())
print(df_join['NumberCarsOwned'].groupby(df_join.BikeBuyer).median())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.Occupation).BikeBuyer.count())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.Gender).BikeBuyer.count())
print(df_join[df_join.BikeBuyer == 1].groupby(df_join.MaritalStatus).BikeBuyer.count())

print(df_join[df_join.BikeBuyer == 1].groupby(df_join.HomeOwnerFlag).BikeBuyer.count())

fig = plt.figure()
fig.suptitle('Distribution of Bike Buyers', fontsize=20)
ax1 = fig.add_subplot(121) #row,#col,fig#
ax1.hist(df_join['YearlyIncome'].groupby(df_join.BikeBuyer).median(), color='red')
ax1.set_title('Non Bike Buyers')
ax1.set_xlabel('# Children At Home')

ax2 = fig.add_subplot(122) #row,#col,fig#
ax2.hist(df_join['NumberCarsOwned'].groupby(df_join.BikeBuyer).median(), color='red')
ax2.set_title('Bike Buyers')
ax2.set_xlabel('# Children At Home')
plt.show()



