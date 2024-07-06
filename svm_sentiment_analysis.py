import pandas as pd
import nltk



import pandas as pd
from termcolor import colored
from sklearn.model_selection import train_test_split

"""
import requests
import json
api_key = "<ENTER-KEY-HERE>"
example_text = "Hollo, wrld" # the text to be spell-checked
endpoint = "https://api.cognitive.microsoft.com/bing/v7.0/SpellCheck"
data = {'text': "hi tere"}
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': api_key,
    }

params = {
    'mkt':'en-us',
    'mode':'proof'
    }
response = requests.post(endpoint, headers=headers, params=params, data=data)

json_response = response.json()
print(json.dumps(json_response, indent=4))
"""

# Define variables
#COLUMNS = ['Sentiment', 'Id', 'Date', 'Flag', 'User', 'Tweet']
#COLUMNS = ['tweet', 'senti']

# Read dataset
dataset = pd.read_csv("C:/Users/Ara-Soft/Desktop/Bachelor's Project/Apple-Twitter-Sentiment-DFE.csv", encoding = 'latin-1')
#print(colored("Columns: {}".format(', '.join(COLUMNS)), "yellow"))
dataset = dataset[dataset.sentiment != "not_relevant"]
# Remove extra columns
print(colored("Useful columns: Sentiment and Tweet", "yellow"))
#print(colored("Removing other columns", "red"))
#dataset.drop(['Id', 'Date', 'Flag', 'User'], axis = 1, inplace = True)
print(colored("Columns removed", "red"))


# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['sentiment'], test_size = 0.20, random_state = 100)
train_dataset = pd.DataFrame({
	'Tweet': X_train,
	'Sentiment': y_train
	})
print(colored("Train data distribution:", "yellow"))
print(train_dataset['Sentiment'].value_counts())
test_dataset = pd.DataFrame({
	'Tweet': X_test,
	'Sentiment': y_test
	})
print(colored("Test data distribution:", "yellow"))
print(test_dataset['Sentiment'].value_counts())
print(colored("Split complete", "yellow"))


# Save train data
print(colored("Saving train data", "yellow"))

train_dataset.to_csv("C:/Users/Ara-Soft/Desktop/Bachelor's Project/train.csv", index = False)
print(colored("Train data saved to data/train.csv", "green"))

# Save test data
print(colored("Saving test data", "yellow"))
test_dataset.to_csv("C:/Users/Ara-Soft/Desktop/Bachelor's Project/test.csv", index = False)
print(colored("Test data saved to data/test.csv", "green"))
# train Data
trainData = pd.read_csv("C:/Users/Ara-Soft/Desktop/Bachelor's Project/train.csv")
#for i in  range(len(trainData)):
 #   trainData.Tweet[i] = str(trainData.Tweet[i])
# test Data
testData = pd.read_csv("C:/Users/Ara-Soft/Desktop/Bachelor's Project/test.csv")
#for i in  range(len(testData)):
 #   testData.Tweet[i] = str(testData.Tweet[i])

from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['Tweet'])
test_vectors = vectorizer.transform(testData['Tweet'])
print(train_vectors)
print(test_vectors)

import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Sentiment'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['Sentiment'], prediction_linear, output_dict=True)
#print('positive: ', report['4'])
#print('negative: ', report['0'])
#print('neutral: ', report['2'])
print(report)
