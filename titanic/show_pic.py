import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_train = pd.read_csv('train.csv')


def show_pic1():
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2,3),(0,0))
    data_train.Survived.value_counts().plot(kind='bar')
    plt.ylabel('number of people')
    plt.title('Sur')

    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind='bar')
    plt.ylabel('Number of people')
    plt.title('Pclass')

    
    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived,data_train.Age)
    plt.ylabel('Age')
    plt.grid(b=True,which='major',axis='y')
    plt.title('Age')

    plt.subplot2grid((2,3),(1,0),colspan=2)
    data_train.Age[data_train.Pclass==1].plot(kind='kde')
    data_train.Age[data_train.Pclass==2].plot(kind='kde')
    data_train.Age[data_train.Pclass==3].plot(kind='kde')
    plt.xlabel('Age')
    plt.xlabel('Density')
    plt.legend(('1st','2nd','3rd'),loc='best') #set our legend for our graph
    plt.title('Age for class')

    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title('Embark number')
    plt.ylabel('Number')

    plt.savefig('pic1.png')
    #plt.show()

def show_class_survival():
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
    df = pd.DataFrame({'sur':Survived_1,'no-sur':Survived_0})
    df.plot(kind='bar',stacked=True)
    plt.title('Class Survival')
    plt.xlabel('Class')
    plt.ylabel('Num')
    plt.savefig('class_survival.png')

if __name__ == '__main__':
    #show_pic1()
    show_class_survival()

