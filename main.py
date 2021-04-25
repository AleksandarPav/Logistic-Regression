import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():
    # the goal is to predict if a person will click on an ad or not

    # loading data into a dataframe
    ad_data = pd.read_csv('advertising.csv')

    # information about data
    print(ad_data.head())
    print(ad_data.info())
    print(ad_data.describe())

    # checking for missing data
    sns.heatmap(ad_data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

    # histogram of the Age
    sns.set_style('whitegrid')
    ad_data['Age'].hist(bins = 30)
    plt.xlabel('Age')

    # jointplot showing Area Income versus Age
    sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)

    # jointplot showing the kde distributions of Daily Time spent on site vs. Age
    sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde', color = 'red')

    # jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
    sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')

    # correlation of all the features, but separated by 'Clicked on Ad' feature
    sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'bwr')

    # splitting the data into training set and testing set
    X = ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp', 'Clicked on Ad'], axis = 1)
    y = ad_data['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    # training and fitting a logistic regression model on the training set
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)

    # predicting values for the testing data
    predictions = logReg.predict(X_test)

    # classification report for the model
    print(classification_report(y_test, predictions))

    plt.show()


if __name__ == '__main__':
    main()