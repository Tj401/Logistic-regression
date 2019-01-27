# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:54:26 2019

@author: kdandebo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sb

F = pd.read_csv("C:/Users/kdandebo/Desktop/Models/Python excercise/Own/train.csv")

print(F.info())
print(F.columns)
#print(F.head(10))
#L = F.head(15)
#print(L)
#print(L.isnull())
#plt.countplot(x='Survived',data=L)
#plt.xlabel('Survived')
#plt.ylabel('Count')
#plt.title('Survival')
#Count1 = 0
#for number in L['Survived']:
#    if number==0:
 #      Count1 = Count1+1
#Count2 = L['Survived'].count() - Count1
 
 
#We can clearly see that the most number of deaths happened from pclass-3 category, becuase they were of low prority than others
sb.countplot(x='Survived',hue = 'Pclass' ,data=F)
 
 #Filling in the missing data in the sheet i.e age, wich is very important from al lthe pclasses, with the average 
 
# we can deduce that 37 - class 1 , 29 from class2 and 24 from class 3 were evident
sb.boxplot(x='Pclass',y='Age',data=F,palette='winter')
 
def Avg_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
 
F['Age'] = F[['Age','Pclass']].apply(Avg_age,axis=1)


#Now the only null attribute is the cabin, we can remove it they are few and not so important

F.drop('Cabin',axis=1,inplace=True)

print(F.columns)

#we need to remove  all the lettered input columns as the algorithm cannot take them as imputs, so replacing them with dummy variables and concatinating with the original data
newsex = pd.get_dummies(F['Sex'],drop_first=True)
newembark = pd.get_dummies(F['Embarked'],drop_first=True)
F.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
F = pd.concat([F,newsex,newembark],axis=1)
print(F.head(5))

#Now we need to train the algorithm with training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(F.drop('Survived',axis=1), 
                                                    F['Survived'], test_size=0.30, 
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

# you can see the accuracy of the report by the f1-score
print(classification_report(y_test,predictions))
