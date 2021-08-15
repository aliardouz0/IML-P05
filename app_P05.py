from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import datetime, timedelta
import seaborn as sns
import timeit
from sklearn.impute import KNNImputer
import statsmodels.formula.api as smf
import statsmodels.api as sm
from nltk.tokenize import RegexpTokenizer
import regex
import re
from bs4 import BeautifulSoup 
from nltk.stem import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
import streamlit as st
import os, urllib, cv2
import urllib
import pickle
import cloudpickle as cp
from urllib.request import urlopen


Nu_SVC_classifier = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/clf.pk"))


#### DEFINE PREPROCESSING FUCTIONS 
##### BEAUTIFUL SOUP
##### Tokenize
##### DELETE STOP WORDS 
##### STEMMER
##### untokenize
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown("# Tags prediction\n Select Run the app\n Write your question and press enter\n")
	#intro_markdown = read_markdown_file("intructions for tha app.md")
    #st.markdown(intro_markdown, unsafe_allow_html=True)

    sw = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/sw.pk"))
    clf = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/clf.pk"))
    tfidf_2 = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/tfidf_2.pk"))
    F_tags = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/F_tags.pk"))
    # Download external dependencies.
    #for filename in EXTERNAL_DEPENDENCIES.keys():
     #   download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":

        readme_text.empty()
        st.code(get_file_content_as_string("app_first_try.py"))
    elif app_mode == "Run the app":
        readme_text.empty() 
        run_the_app()
        

        
def run_the_app():
    with open('sw.pk', 'rb') as file:
        sw = pickle.load(file)
    with open('clf.pk', 'rb') as file:
        clf = pickle.load(file)
    with open('tfidf_2.pk', 'rb') as file:
        tfidf_2 = pickle.load(file)
    with open('F_tags.pk', 'rb') as file:
        F_tags = pickle.load(file)
        
        
    sentence = st.text_input('Input your sentence here:') 
    X = tfidf_2.fit_transform(preprocessing_text(sentence))
    tf_idf = pd.DataFrame(data = X.toarray(), columns=tfidf_2.get_feature_names())
    prediction = pd.DataFrame(clf.predict(tf_idf), columns = F_tags)
    tags=[]
    for col in prediction.columns:
        if prediction[col][0]==1:
            tags.append(col)
    if sentence:
        st.write(tags)

def tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(regex.sub(u'[^\p{Latin}]', u' ', text))
	return tokens


def apply_BSoup(body):       



	soup = BeautifulSoup(body, 'html.parser')
	text = soup.get_text() 
	#souped_body.append(text)
		
	return text

def delete_sw(body):
    with open('sw.pk', 'rb') as file:
        sw = pickle.load(file)		
    filtered_body=[]
    
    x = tokenize(body)
    filtered_words = [word for word in x if word not in list(sw)]
    filtered_body.append(filtered_words)
		
    return filtered_body


def SB_stemmer(body):
    with open('sw.pk', 'rb') as file:
        sw = pickle.load(file)		
    snowball = SnowballStemmer(language='english')

    stemmed_body=[]
    x = tokenize(body)

    filtered_words = [snowball.stem(word) for word in x if word not in list(sw)]
    stemmed_body.append(filtered_words)
		
    return(filtered_words)

def untokenize(body):
	untokenized_body = []
	untokenized_body.append(TreebankWordDetokenizer().detokenize(body))
		
	return untokenized_body

def preprocessing_text(text):
    output = untokenize(SB_stemmer(apply_BSoup(text)))
    return output

from pathlib import Path
import streamlit as st

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def get_file_content_as_string(path):
    url = 'https://github.com/aliardouz0/IML/blob/29f080131c22971519363acb79676a766881c9ff/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")
            
            

if __name__ == "__main__":
    main()