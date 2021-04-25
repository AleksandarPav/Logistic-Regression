# Logistic Regression
 
A fake advertising data set is given, indicating whether or not a particular internet user clicked on an Advertisement on a company website. Logisti regression model is created that predicts whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

'Daily Time Spent on Site': consumer time on site in minutes
'Age': cutomer age in years
'Area Income': Avg. income of geographical area of consumer
'Daily Internet Usage': Avg. minutes a day consumer is on the internet
'Ad Topic Line': Headline of the advertisement
'City': City of consumer
'Male': Whether or not consumer was male
'Country': Country of consumer
'Timestamp': Time at which consumer clicked on Ad or closed window
'Clicked on Ad': 0 or 1 indicated clicking on Ad

First, existance of missing data is checked. Correlation between Area Income and Age, Daily Time Spent on Site and Age, Daily Time Spent on Site and Daily Internet Usage and, finally, of every feature with every other feature is examined. Data is then splitted to training set and test set, where test set size is a third of all the data. Logistic regression model is created and fitted to the training set. Predictions are done on the test set. At the end, classification report is created to evaluate performance of the model.
