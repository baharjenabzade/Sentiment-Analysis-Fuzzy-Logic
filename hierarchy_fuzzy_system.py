

import pandas as pd
import re
import numpy as np
import skfuzzy as fuzz

import csv

import matplotlib.pyplot as plt
from termcolor import colored

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import sentiwordnet as swn


from sklearn.preprocessing import LabelEncoder,StandardScaler,QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('sentiwordnet')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.corpus import wordnet as wn
import ssl
import re


from pycorenlp import StanfordCoreNLP
"""
nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("I love you. I hate him. You are nice. He is dumb",
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
print(res)
"""

import time
start = time.time()



data = pd.read_csv("Tweets.csv", encoding='ISO-8859-1')  
print(data)
doc = data.selected_text
sentiment = data.sentiment



tweets = []
senti = []
for i in range(len(doc)):
    s1 = str(doc[i])
    s2 = s1.lower()
    tweets.append(s2)   
    senti.append(sentiment[i])

# text pre-processing 
def decontracted(phrase):   
        
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"@", "" , phrase)        
        phrase =  re.sub(r"http\S+", "", phrase)   
        phrase = re.sub(r"#", "", phrase)          
    
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

for i in range(len(doc)):
    tweets[i] = decontracted(tweets[i])
         
vader = SentimentIntensityAnalyzer()
afinn = Afinn()

def first_fuzzy_system(tweets, senti):

    vader_input_pos = np.arange(0, 1, 0.1)
    vader_input_neg = np.arange(0, 1, 0.1)
    first_fuzzy_out = np.arange(0, 10, 1)

    #triangular membership function for low positive input
    pos_low = fuzz.trapmf(vader_input_pos, [0,0,0.4,0.6])
    #trapezoid membership function for medium positive input
    pos_med = fuzz.trapmf(vader_input_pos, [0,0.4,0.5,1])
    #trapezoid membership function for high positive input
    pos_high = fuzz.trapmf(vader_input_pos, [0.5,0.55,1,1])
    #triangular membership function for low negative input
    neg_low = fuzz.trapmf(vader_input_neg, [0,0,0.2,0.3])
    #trapezoid membership function for medium negative input
    neg_med = fuzz.trapmf(vader_input_neg, [0,0.3,0.5,1])
    #trapezoid membership function for high negative input
    neg_high = fuzz.trapmf(vader_input_neg, [0.5,0.55,1,1])


    op_neg = fuzz.trapmf(first_fuzzy_out, [0,0,0,5])
    op_neu = fuzz.trapmf(first_fuzzy_out, [0,5,5,10])
    op_pos = fuzz.trapmf(first_fuzzy_out, [5,5,5,10])
    
    sentiment = []
    sentiment_polarity = []
    for i in range(len(doc)):
        vader_score = vader.polarity_scores(tweets[i])
        print(vader_score)
        posscore = afinn.score(tweets[i])
        posscore = vader_score['pos']
        negscore = vader_score['neg']
        compoundscore = vader_score['compound']

        print("\nPositive Score for each  tweet :") 
        if (posscore == 1):
            posscore = 0.9 
        else:
            posscore = round(posscore,1)
            if posscore == 1:
                posscore = 0.9
        print(posscore)

        print("\nNegative Score for each  tweet :")
        if (negscore == 1):
            negscore = 0.9
        else:
            negscore = round(negscore,1)
            if negscore == 1:
                negscore = 0.9
        print(negscore)

        pos_level_low = fuzz.interp_membership(vader_input_pos, pos_low, posscore)
        pos_level_med  = fuzz.interp_membership(vader_input_pos, pos_med, posscore)
        pos_level_high = fuzz.interp_membership(vader_input_pos, pos_high, posscore)
        
        neg_level_low = fuzz.interp_membership(vader_input_neg, neg_low, negscore)
        neg_level_med = fuzz.interp_membership(vader_input_neg, neg_med, negscore)
        neg_level_high = fuzz.interp_membership(vader_input_neg, neg_high, negscore)

        active_rule1 = np.fmin(pos_level_low, neg_level_low)
        active_rule2 = np.fmin(pos_level_med, neg_level_low)
        active_rule3 = np.fmin(pos_level_high, neg_level_low)
        active_rule4 = np.fmin(pos_level_low, neg_level_med)
        active_rule5 = np.fmin(pos_level_med, neg_level_med)
        active_rule6 = np.fmin(pos_level_high, neg_level_med)
        active_rule7 = np.fmin(pos_level_low, neg_level_high)
        active_rule8 = np.fmin(pos_level_med, neg_level_high)
        active_rule9 = np.fmin(pos_level_high, neg_level_high)

        neg = np.fmax(np.fmax(active_rule4, active_rule7), active_rule8)    
        op_activation_low = np.fmin(neg,op_neg)
        
        neu = np.fmax(np.fmax(active_rule1,active_rule5), active_rule9)
             
        op_activation_med = np.fmin(neu,op_neu)
        
        pos = np.fmax(np.fmax(active_rule2,active_rule3), active_rule6) 
        op_activation_high = np.fmin(pos,op_pos)
        op0 = np.zeros_like(first_fuzzy_out)


        aggregated = np.fmax(op_activation_low, np.fmax(op_activation_med, op_activation_high))
    
        # Calculate defuzzified result
        op = fuzz.defuzz(first_fuzzy_out, aggregated, 'som')
        output=round(op,2)
        op_activation = fuzz.interp_membership(first_fuzzy_out, aggregated, op)  # for plot

        #Visualize Aggregated Membership
        fig, ax0 = plt.subplots(figsize=(8, 3))
    
        ax0.plot(first_fuzzy_out, op_neg, 'b', linewidth=0.5, linestyle='--',label= 'Negative')
        ax0.plot(first_fuzzy_out, op_neu, 'g', linewidth=0.5, linestyle='--',label= 'Neutral')
        ax0.plot(first_fuzzy_out, op_pos, 'r', linewidth=0.5, linestyle='--',label= 'Positive')
        ax0.fill_between(first_fuzzy_out, op0, aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([op, op], [0, op_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Aggregated membership and result (line)')
        ax0.legend()
        
    #    # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
    #    
        plt.tight_layout()
        
        fig, ax0 = plt.subplots(figsize=(8, 3))
        
        ax0.fill_between(first_fuzzy_out, op0, op_activation_low, facecolor='b', alpha=0.7)
        ax0.plot(first_fuzzy_out, op_neg, 'b', linewidth=0.5, linestyle='--',label= 'Negative' )
        ax0.fill_between(first_fuzzy_out, op0, op_activation_med, facecolor='g', alpha=0.7)
        ax0.plot(first_fuzzy_out, op_neu, 'g', linewidth=0.5, linestyle='--', label='Neutral')
        ax0.fill_between(first_fuzzy_out, op0, op_activation_high, facecolor='r', alpha=0.7)
        ax0.plot(first_fuzzy_out, op_pos, 'r', linewidth=0.5, linestyle='--', label='Positive')
        ax0.plot([op, op], [0, op_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Output membership activity')
        ax0.legend()

        if 0<=(output)<=3.33:    # R
            print("\nOutput after Defuzzification: Negative")
            sentiment.append('negative')
            sentiment_polarity.append(output)
        
        elif 3.34<=(output)<=6.66:
            print("\nOutput after Defuzzification: Neutral")
            sentiment.append("neutral")
            sentiment_polarity.append(output)
    
        elif 6.66<(output)<=10:
            print("\nOutput after Defuzzification: Positive")
            sentiment.append('positive')
            sentiment_polarity.append(output)
            
        print("Doc sentiment: " +str(senti[i])+"\n")  
    y_true = senti
    y_pred = sentiment
    
    a1 = accuracy_score(y_true, y_pred)  

    print("Accuracy score (MACRO): " + str(round((a1*100),2)))

    p1 = precision_score(y_true, y_pred, average='macro')  

    print("Precision score (MACRO): " + str(round((p1*100),2)))

    r1 = recall_score(y_true, y_pred, average='macro')  

    print("Recall score (MACRO): " + str(round((r1*100),2)))


    return sentiment, sentiment_polarity

senti1, polar1 = first_fuzzy_system(tweets, senti)

def cal_afinn_score(tweet):
    pos_score = 0
    neg_score = 0
    tokens = word_tokenize(tweet)
    for i in range(len(tokens)):
        afinn_score = afinn.score(tokens[i])
        print(afinn_score)
        if afinn_score > 0:
            pos_score += afinn_score
        if afinn_score < 0:
            neg_score += -(afinn_score)
    pos_score /= (len(tokens) + 1)
    neg_score /= (len(tokens) + 1)
    return pos_score , neg_score



def second_fuzzy_system(tweets, senti):
    vader_input_pos = np.arange(0, 1, 0.1)
    vader_input_neg = np.arange(0, 1, 0.1)
    second_fuzzy_out = np.arange(0, 10, 1)

    #triangular membership function for low positive input
    pos_low = fuzz.trapmf(vader_input_pos, [0,0,0.3,0.5])
    #trapezoid membership function for medium positive input
    pos_med = fuzz.trapmf(vader_input_pos, [0,0.4,0.5,1])
    #trapezoid membership function for high positive input
    pos_high = fuzz.trapmf(vader_input_pos, [0.5,0.55,1,1])
    #triangular membership function for low negative input
    neg_low = fuzz.trapmf(vader_input_neg, [0,0,0.2,0.3])
    #trapezoid membership function for medium negative input
    neg_med = fuzz.trapmf(vader_input_neg, [0,0.3,0.5,1])
    #trapezoid membership function for high negative input
    neg_high = fuzz.trapmf(vader_input_neg, [0.5,0.55,1,1])


    op_neg = fuzz.trimf(second_fuzzy_out, [0,0,4])
    op_neu = fuzz.trimf(second_fuzzy_out, [0,5,10])
    op_pos = fuzz.trimf(second_fuzzy_out, [4,10,10])
    sentiment = []
    sentiment_polarity = []
    for i in range(len(doc)):

        posscore , negscore = cal_afinn_score(tweets[i])
        """
        afinn_score = afinn.score(tweets[i])
        
        if afinn_score > 10:
            afinn_score = 10
        elif afinn_score < -10:
            afinn_score = -10
        if afinn_score > 0 :
            if afinn_score > 5:
                posscore = afinn_score
                negscore = 10 - posscore
            if afinn_score < 5:
                negscore = afinn_score
                posscore = 10 - negscore
            if afinn_score == 5:
                posscore = afinn_score
                negscore = 0


        elif afinn_score < 0 :
            if afinn_score > -5:
                posscore = -(afinn_score)
                negscore = 10 - posscore
            if afinn_score < -5:
                negscore = -(afinn_score)
                posscore = 10 - negscore
            if afinn_score == -5:
                negscore = -(afinn_score)
                posscore = 0
        else:
            posscore = negscore = 5
        posscore = posscore / 10
        negscore = negscore / 10
        print("afinn scores" , afinn_score, posscore , negscore)
        """
        print("\nPositive Score for each  tweet :") 
        if (posscore >= 1):
            posscore = 0.9 
        else:
            posscore = round(posscore,1)
            if posscore == 1:
                posscore = 0.9
        print(posscore)

        print("\nNegative Score for each  tweet :")
        if (negscore >= 1):
            negscore = 0.9
        else:
            negscore = round(negscore,1)
            if negscore == 1:
                negscore = 0.9
        print(negscore)

        pos_level_low = fuzz.interp_membership(vader_input_pos, pos_low, posscore)
        pos_level_med  = fuzz.interp_membership(vader_input_pos, pos_med, posscore)
        pos_level_high = fuzz.interp_membership(vader_input_pos, pos_high, posscore)
        
        neg_level_low = fuzz.interp_membership(vader_input_neg, neg_low, negscore)
        neg_level_med = fuzz.interp_membership(vader_input_neg, neg_med, negscore)
        neg_level_high = fuzz.interp_membership(vader_input_neg, neg_high, negscore)

        active_rule1 = np.fmin(pos_level_low, neg_level_low)
        active_rule2 = np.fmin(pos_level_med, neg_level_low)
        active_rule3 = np.fmin(pos_level_high, neg_level_low)
        active_rule4 = np.fmin(pos_level_low, neg_level_med)
        active_rule5 = np.fmin(pos_level_med, neg_level_med)
        active_rule6 = np.fmin(pos_level_high, neg_level_med)
        active_rule7 = np.fmin(pos_level_low, neg_level_high)
        active_rule8 = np.fmin(pos_level_med, neg_level_high)
        active_rule9 = np.fmin(pos_level_high, neg_level_high)

        neg = np.fmax(np.fmax(active_rule4, active_rule7), active_rule8)    
        op_activation_low = np.fmin(neg,op_neg)
        
        neu = np.fmax(np.fmax(active_rule1,active_rule5), active_rule9)
             
        op_activation_med = np.fmin(neu,op_neu)
        
        pos = np.fmax(np.fmax(active_rule2,active_rule3), active_rule6) 
        op_activation_high = np.fmin(pos,op_pos)


        aggregated = np.fmax(op_activation_low, np.fmax(op_activation_med, op_activation_high))
    
        # Calculate defuzzified result
        op = fuzz.defuzz(second_fuzzy_out, aggregated, 'som')
        output=round(op,2)

        if 0<=(output)<=3.33:    # R
            print("\nOutput after Defuzzification: Negative")
            sentiment.append('negative')
            sentiment_polarity.append(output)
        
        elif 3.34<=(output)<=6.66:
            print("\nOutput after Defuzzification: Neutral")
            sentiment.append("neutral")
            sentiment_polarity.append(output)
    
        elif 6.66<(output)<=10:
            print("\nOutput after Defuzzification: Positive")
            sentiment.append('positive')
            sentiment_polarity.append(output)
            
        print("Doc sentiment: " +str(senti[i])+"\n")  
    y_true = senti
    y_pred = sentiment
    
    a1 = accuracy_score(y_true, y_pred)  

    print("Accuracy score : " + str(round((a1*100),2)))

    p1 = precision_score(y_true, y_pred, average='macro')  

    print("Precision score (MACRO): " + str(round((p1*100),2)))

    r1 = recall_score(y_true, y_pred, average='macro')  

    print("Recall score (MACRO): " + str(round((r1*100),2)))

    return sentiment , sentiment_polarity

#senti2, polar2 = second_fuzzy_system(tweets, senti)



def third_fuzzy_system(tweets, senti):
    vader_input_pos = np.arange(0, 1, 0.1)
    vader_input_neg = np.arange(0, 1, 0.1)
    third_fuzzy_out = np.arange(0, 10, 1)

    #triangular membership function for low positive input
    pos_low = fuzz.trapmf(vader_input_pos, [0,0,0.1,0.1])
    #trapezoid membership function for medium positive input
    pos_med = fuzz.trapmf(vader_input_pos, [0,0.4,0.5,1])
    #trapezoid membership function for high positive input
    pos_high = fuzz.trapmf(vader_input_pos, [0.4,0.45,1,1])
    #triangular membership function for low negative input
    neg_low = fuzz.trapmf(vader_input_neg, [0,0,0.1,0.5])
    #trapezoid membership function for medium negative input
    neg_med = fuzz.trapmf(vader_input_neg, [0,0.4,0.4,1])
    #trapezoid membership function for high negative input
    neg_high = fuzz.trapmf(vader_input_neg, [0.4,0.55,1,1])


    op_neg = fuzz.trimf(third_fuzzy_out, [0,0,4])
    op_neu = fuzz.trimf(third_fuzzy_out, [0,5,10])
    op_pos = fuzz.trimf(third_fuzzy_out, [4,10,10])
    sentiment = []
    sentiment_polarity = []
    for i in range(len(doc)):
        textblob_score = TextBlob(tweets[i]).sentiment.polarity
        """
        sent           = TextBlob(tweets[i], analyzer = NaiveBayesAnalyzer())
        
        posscore     = sent.sentiment.p_pos
        negscore     = sent.sentiment.p_neg
        print(posscore, negscore , "score")
        """
                                               
        if textblob_score < 0 :
            if textblob_score > -0.5:
                posscore = -(textblob_score)
                negscore = 1 - posscore
            if textblob_score < -0.5:
                negscore = -(textblob_score)
                posscore = 1 - negscore
            if textblob_score == -0.5:
                negscore = -(textblob_score)
                posscore = 0 
        elif textblob_score > 0 :
            if textblob_score > 0.5:
                posscore = textblob_score
                negscore = 1 - posscore
            if textblob_score < 0.5:
                negscore = textblob_score
                posscore = 1 - negscore
            if textblob_score == 0.5:
                posscore = textblob_score
                negscore = 0     
        else:
            posscore = negscore = 0
    
        print("textblob scores" , textblob_score, posscore , negscore)
        
        print("\nPositive Score for each  tweet :") 
        if (posscore == 1):
            posscore = 0.9 
        else:
            posscore = round(posscore,1)
            if posscore == 1:
                posscore = 0.9
        print(posscore)

        print("\nNegative Score for each  tweet :")
        if (negscore == 1):
            negscore = 0.9
        else:
            negscore = round(negscore,1)
            if negscore == 1:
                negscore = 0.9
        print(negscore)

        pos_level_low = fuzz.interp_membership(vader_input_pos, pos_low, posscore)
        pos_level_med  = fuzz.interp_membership(vader_input_pos, pos_med, posscore)
        pos_level_high = fuzz.interp_membership(vader_input_pos, pos_high, posscore)
        
        neg_level_low = fuzz.interp_membership(vader_input_neg, neg_low, negscore)
        neg_level_med = fuzz.interp_membership(vader_input_neg, neg_med, negscore)
        neg_level_high = fuzz.interp_membership(vader_input_neg, neg_high, negscore)

        active_rule1 = np.fmin(pos_level_low, neg_level_low)
        active_rule2 = np.fmin(pos_level_med, neg_level_low)
        active_rule3 = np.fmin(pos_level_high, neg_level_low)
        active_rule4 = np.fmin(pos_level_low, neg_level_med)
        active_rule5 = np.fmin(pos_level_med, neg_level_med)
        active_rule6 = np.fmin(pos_level_high, neg_level_med)
        active_rule7 = np.fmin(pos_level_low, neg_level_high)
        active_rule8 = np.fmin(pos_level_med, neg_level_high)
        active_rule9 = np.fmin(pos_level_high, neg_level_high)

        neg = np.fmax(np.fmax(active_rule4, active_rule7), active_rule8)    
        op_activation_low = np.fmin(neg,op_neg)
        
        neu = np.fmax(np.fmax(active_rule1,active_rule5), active_rule9)
             
        op_activation_med = np.fmin(neu,op_neu)
        
        pos = np.fmax(np.fmax(active_rule2,active_rule3), active_rule6) 
        op_activation_high = np.fmin(pos,op_pos)


        aggregated = np.fmax(op_activation_low, np.fmax(op_activation_med, op_activation_high))
    
        # Calculate defuzzified result
        op = fuzz.defuzz(third_fuzzy_out, aggregated, 'som')
        output=round(op,2)

        if 0<=(output)<=3.33:    # R
            print("\nOutput after Defuzzification: Negative")
            sentiment.append('negative')
            sentiment_polarity.append(output)
        
        elif 3.34<=(output)<=6.66:
            print("\nOutput after Defuzzification: Neutral")
            sentiment.append("neutral")
            sentiment_polarity.append(output)
    
        elif 6.66<(output)<=10:
            print("\nOutput after Defuzzification: Positive")
            sentiment.append('positive')
            sentiment_polarity.append(output)
            
        print("Doc sentiment: " +str(senti[i])+"\n")  
    y_true = senti
    y_pred = sentiment
    
    a1 = accuracy_score(y_true, y_pred)  

    print("Accuracy score : " + str(round((a1*100),2)))

    p1 = precision_score(y_true, y_pred, average='macro')  

    print("Precision score (MACRO): " + str(round((p1*100),2)))

    r1 = recall_score(y_true, y_pred, average='macro')  

    print("Recall score (MACRO): " + str(round((r1*100),2)))
    return sentiment, sentiment_polarity

#senti3, polar3 = third_fuzzy_system(tweets, senti)

"""
#Edits After Removing Stopwords
Edited_Review = data['text'].copy()
data['raw_text'] = Edited_Review

# Function to preprocess Reviews data
def preprocess_Reviews_data(data,name):
    # Proprocessing the data
    data[name]=data[name].str.lower()
    # Code to remove the Hashtags from the text
    data[name]=data[name].apply(lambda x: str(x))
    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+','',x))
    # Code to remove the links from the text
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Code to remove the Special characters from the text 
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Code to substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    #data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    # Remove the twitter handlers
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))

def rem_stopwords_tokenize(data,name):
      
    def getting(sen):
        example_sent = sen
        
        filtered_sentence = [] 

        stop_words = set(stopwords.words('english')) 

        word_tokens = word_tokenize(example_sent) 
        
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        return filtered_sentence
    # Using "getting(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x

lemmatizer = WordNetLemmatizer()
def Lemmatization(data,name):
    def getting2(sen):
        
        example = sen
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        
        # Remove characters which have length less than 2  
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        
        return cleaned_data_title
    # Using "getting2(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting2(i))
    data[name]=x

def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))


# Using the preprocessing function to preprocess the hotel data
preprocess_Reviews_data(data,'raw_text')
# Using tokenizer and removing the stopwords
rem_stopwords_tokenize(data,'raw_text')
# Converting all the texts back to sentences
make_sentences(data,'raw_text')

#Edits After Lemmatization
final_Edit = data['raw_text'].copy()
data["after_lemmatization"] = final_Edit

# Using the Lemmatization function to lemmatize the hotel data
Lemmatization(data,'after_lemmatization')
# Converting all the texts back to sentences
make_sentences(data,'after_lemmatization')


pos=neg=obj=count=0

postagging = []

for review in data['after_lemmatization']:
    list = word_tokenize(review)
    postagging.append(nltk.pos_tag(list))

data['pos_tags'] = postagging

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    #Lemmatization
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    #Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. 
    #Synset instances are the groupings of synonymous words that express the same concept. 
    #Some of the words have only one Synset and some have several.
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

    pos=neg=obj=count=0
    
    ###################################################################################
senti_score = []
pos_score = []
neg_score = []
for pos_val in data['pos_tags']:
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    for score in senti_val:
        try:
            pos = pos + score[1]  #positive score is stored at 2nd position
            neg = neg + score[2]  #negative score is stored at 3rd position
        except:
            continue
    senti_score.append(pos - neg)
    pos_score.append(pos)
    neg_score.append(neg)
    #print(pos , neg)
    pos=neg=0    
    
data['senti_score'] = senti_score


print(data.head(50))
"""


def fourth_fuzzy_system(tweets, senti):

    vader_input_pos = np.arange(0, 1, 0.1)
    vader_input_neg = np.arange(0, 1, 0.1)
    fourth_fuzzy_out = np.arange(0, 10, 1)

    #triangular membership function for low positive input
    pos_low = fuzz.trapmf(vader_input_pos, [0,0,0.3,0.4])
    #trapezoid membership function for medium positive input
    pos_med = fuzz.trapmf(vader_input_pos, [0,0.3,0.5,1])
    #trapezoid membership function for high positive input
    pos_high = fuzz.trapmf(vader_input_pos, [0.5,0.55,1,1])
    #triangular membership function for low negative input
    neg_low = fuzz.trapmf(vader_input_neg, [0,0,0.1,0.4])
    #trapezoid membership function for medium negative input
    neg_med = fuzz.trapmf(vader_input_neg, [0,0.3,0.5,1])
    #trapezoid membership function for high negative input
    neg_high = fuzz.trapmf(vader_input_neg, [0.5,0.55,1,1])


    op_neg = fuzz.trimf(fourth_fuzzy_out, [0,0,4])
    op_neu = fuzz.trimf(fourth_fuzzy_out, [0,5,10])
    op_pos = fuzz.trimf(fourth_fuzzy_out, [4,10,10])
    sentiment = []
    sentiment_polarity = []
    for i in range(len(doc)):
        sentiwordnet_score = data["senti_score"][i]

        posscore = pos_score[i]
        negscore = neg_score[i]
        """
        if sentiwordnet_score < 0 :
            if sentiwordnet_score > -0.5:
                posscore = -(sentiwordnet_score)
                negscore = 1 - posscore
            if sentiwordnet_score < -0.5:
                negscore = -(sentiwordnet_score)
                posscore = 1 - negscore
            if sentiwordnet_score == -0.5:
                negscore = -(sentiwordnet_score)
                posscore = 0 
        elif sentiwordnet_score > 0 :
            if sentiwordnet_score > 0.5:
                posscore = sentiwordnet_score
                negscore = 1 - posscore
            if sentiwordnet_score < 0.5:
                negscore = sentiwordnet_score
                posscore = 1 - negscore
            if sentiwordnet_score == 0.5:
                posscore = sentiwordnet_score
                negscore = 0     
        else:
            posscore = negscore = 0
    
        print("sentiwordnet scores" , sentiwordnet_score, posscore , negscore)
        """
        print("\nPositive Score for each  tweet :") 
        if (posscore >= 1):
            posscore = 0.9 
        else:
            posscore = round(posscore,1)
            if posscore == 1:
                posscore = 0.9
        print(posscore)

        print("\nNegative Score for each  tweet :")
        if (negscore >= 1):
            negscore = 0.9
        else:
            negscore = round(negscore,1)
            if negscore == 1:
                negscore = 0.9
        print(negscore)

        pos_level_low = fuzz.interp_membership(vader_input_pos, pos_low, posscore)
        pos_level_med  = fuzz.interp_membership(vader_input_pos, pos_med, posscore)
        pos_level_high = fuzz.interp_membership(vader_input_pos, pos_high, posscore)
        
        neg_level_low = fuzz.interp_membership(vader_input_neg, neg_low, negscore)
        neg_level_med = fuzz.interp_membership(vader_input_neg, neg_med, negscore)
        neg_level_high = fuzz.interp_membership(vader_input_neg, neg_high, negscore)

        active_rule1 = np.fmin(pos_level_low, neg_level_low)
        active_rule2 = np.fmin(pos_level_med, neg_level_low)
        active_rule3 = np.fmin(pos_level_high, neg_level_low)
        active_rule4 = np.fmin(pos_level_low, neg_level_med)
        active_rule5 = np.fmin(pos_level_med, neg_level_med)
        active_rule6 = np.fmin(pos_level_high, neg_level_med)
        active_rule7 = np.fmin(pos_level_low, neg_level_high)
        active_rule8 = np.fmin(pos_level_med, neg_level_high)
        active_rule9 = np.fmin(pos_level_high, neg_level_high)

        neg = np.fmax(np.fmax(active_rule4, active_rule7), active_rule8)    
        op_activation_low = np.fmin(neg,op_neg)
        
        neu = np.fmax(np.fmax(active_rule1,active_rule5), active_rule9)
             
        op_activation_med = np.fmin(neu,op_neu)
        
        pos = np.fmax(np.fmax(active_rule2,active_rule3), active_rule6) 
        op_activation_high = np.fmin(pos,op_pos)


        aggregated = np.fmax(op_activation_low, np.fmax(op_activation_med, op_activation_high))
    
        # Calculate defuzzified result
        op = fuzz.defuzz(fourth_fuzzy_out, aggregated, 'som')
        output=round(op,2)

        if 0<=(output)<=3.33:    # R
            print("\nOutput after Defuzzification: Negative")
            sentiment.append('negative')
            sentiment_polarity.append(output)
        
        elif 3.34<=(output)<=6.66:
            print("\nOutput after Defuzzification: Neutral")
            sentiment.append("neutral")
            sentiment_polarity.append(output)
    
        elif 6.66<(output)<=10:
            print("\nOutput after Defuzzification: Positive")
            sentiment.append('positive')
            sentiment_polarity.append(output)
            
        print("Doc sentiment: " +str(senti[i])+"\n")  
    y_true = senti
    y_pred = sentiment
    
    a1 = accuracy_score(y_true, y_pred)  

    print("Accuracy score : " + str(round((a1*100),2)))

    p1 = precision_score(y_true, y_pred, average='macro')  

    print("Precision score (MACRO): " + str(round((p1*100),2)))

    r1 = recall_score(y_true, y_pred, average='macro')  

    print("Recall score (MACRO): " + str(round((r1*100),2)))
    return sentiment, sentiment_polarity

#senti4, polar4 = fourth_fuzzy_system(tweets, senti)

def match_word(text_list, match):
    index = 0
    polar_list = [0]* len(text_list)
    for item in text_list:
        if item == match:
            polar_list[index] = 1
            index += 1
        else:
            polar_list[index] = 0
            index += 1
    return polar_list
def write_in_csv():
    # field names 
    fields = ['Fuzzy system 1', 'Fuzzy system 2', 'Fuzzy system 3' , 'Fuzzy system 4'] 
    senti1, polar1 = first_fuzzy_system(tweets, senti)
    senti2, polar2 = second_fuzzy_system(tweets, senti)
    senti3, polar3 = third_fuzzy_system(tweets, senti)
    senti4, polar4 = fourth_fuzzy_system(tweets, senti)
    # data rows of csv file 
    rows = []
    for i in range(len(tweets)):
        item = [senti1[i], senti2[i], senti3[i], senti4[i]]
        rows.append(item) 
        
    # name of csv file 
    filename = "FuzzyRecords.csv"
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)
    return
#write_in_csv()
# if its possible add lstm + fuzzy model too
def fuzzy_system_learning(sentiment):
    data = pd.read_csv("FuzzyRecords.csv" ,  encoding='ISO-8859-1')
    senti1 = data["Fuzzy system 1"]
    senti2 = data["Fuzzy system 2"]
    senti3 = data["Fuzzy system 3"]
    senti4 = data["Fuzzy system 4"]

    epsilon = 0.001
    weights = [1,1,1,1]
    polar_list = []
    tuned_senti = []
    for i in range(len(doc)):
        true_senti = sentiment[i]
        polar_list.append([senti1[i] , senti2[i] , senti3[i], senti4[i]])
        print("polar bef" , polar_list[i])
        list1 = match_word(polar_list[i], "positive")
        print("list1 bef" , list1)
        for k in range(len(list1)):
            if list1[k] == 1:
                list1[k] = 1 * weights[k]
        sum1 = sum(list1)
        list2 = match_word(polar_list[i], "neutral")
        for k in range(len(list2)):
            if list2[k] == 1:
                list2[k] = 1 * weights[k]
        sum2 = sum(list2)
        list3 = match_word(polar_list[i], "negative")
        for k in range(len(list3)):
            if list3[k] == 1:
                list3[k] = 1 * weights[k]
        sum3 = sum(list3)
        max_sum = np.array([sum1, sum2, sum3])
        print(list1, list2, list3, max_sum )
        max_val = max_sum.max()
        if max_val == sum1:
            tuned_senti.append('positive')
        elif max_val == sum2:
            tuned_senti.append('neutral')
        elif max_val == sum3:
            tuned_senti.append('negative')
        print(max_val, "max")
        print("senti", sentiment[i])
        print("tuned", tuned_senti[i])
        print("weights", weights)
        if sum1 == max_val:
            if sentiment[i] == 'positive':
                print("polar pos" , polar_list[i])
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'positive':
                        weights[j] += 0.0001
                    # should wrong ones be decreased?
                    elif polar_list[i][j] != 'positive':
                        weights[j] -= 0.0001
            elif sentiment[i] != 'positive':
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'positive'or polar_list[i][j] != sentiment[i]:
                        weights[j] -= 0.0001
                    if polar_list[i][j] != 'positive' and polar_list[i][j] == sentiment[i]:
                        weights[j] += 0.0001

        if sum2 == max_val:
            # what if it wasnt
            if sentiment[i] == 'neutral':
                print("polar neu" , polar_list[i])
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'neutral':
                        weights[j] += 0.0001
                    elif polar_list[i][j] != 'neutral':
                        weights[j] -= 0.0001
            elif sentiment[i] != 'neutral':
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'neutral'or polar_list[i][j] != sentiment[i]:
                        weights[j] -= 0.0001
                    if polar_list[i][j] != 'neutral' and polar_list[i][j] == sentiment[i]:
                        weights[j] += 0.0001
        if sum3 == max_val:
            if sentiment[i] == 'negative':
                print("polar neg" , polar_list[i])
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'negative':
                        weights[j] += 0.0001
                    elif polar_list[i][j] != 'negative':
                        weights[j] -= 0.0001
            elif sentiment[i] != 'negative':
                for j in range(len(polar_list[i])):
                    if polar_list[i][j] == 'negative'or polar_list[i][j] != sentiment[i]:
                        weights[j] -= 0.0001
                    if polar_list[i][j] != 'negative'and polar_list[i][j] == sentiment[i]:
                        weights[j] += 0.0001
    print(weights)
    #print(tuned_senti)
    acc = accuracy_score(tuned_senti, sentiment)
    print("Accuracy :", acc)


    return
#fuzzy_system_learning(sentiment)

