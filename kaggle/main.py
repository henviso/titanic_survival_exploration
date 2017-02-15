import math

# Import libraries necessary for this project
import numpy as np
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

# Load the dataset
in_file = 'test.csv'
full_data = pd.read_csv(in_file)
#print full_data

# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
#display(outcomes.head())
data = full_data

# Show the new dataset with 'Survived' removed
display(data.head())

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
def predict_woman(woman):
    age = woman.Age
    pclass = woman.Pclass
    prediction = 1
    if (pclass == 1 and age < 10) or (pclass == 3 and (age <= 50 and age >= 20)):
        prediction = 0
    #print 'Prediction for woman with Age ' + str(age) + ' and Pclass ' + str(pclass) + " = " + str(prediction)
    return prediction

def predict_man(man):
    #print 'Prediction man ' + str(man)
    age = man.Age
    pclass = man.Pclass
    prediction = 0
    if not(age is None) and not(math.isnan(age)) and ((pclass != 3 and age < 10) or (pclass == 1 and (age >= 30 and age < 40))):
            prediction = 1
    #print 'Prediction for man with Age ' + str(age) + ' and Pclass ' + str(pclass) + " = " + str(prediction)
    return prediction

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        sex = passenger['Sex']
        if sex == 'male':
            predictions.append(predict_man(passenger))
        else:
            predictions.append(predict_woman(passenger))
        #print 'Sex ' + passenger['Sex'] + ' Pred ' + str(int(passenger['Sex'] == 'female'))
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
display(predictions)

ans = pd.DataFrame({'PassengerId' : full_data['PassengerId'], 'Survived' : predictions});
display(ans)
ans.to_csv('result.csv', mode = 'w', index=False)


