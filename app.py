from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

app=Flask(__name__)
pickle_in=open('tfidf.pkl','rb')
tfidf_pkl=pickle.load(pickle_in)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        text=str(request.form['review'])
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', text)

        # remove all single characters
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        #print(processed_feature)


        #creating array from list
        text=np.array([processed_feature])
        #text

        # Create new tfidfVectorizer with old vocabulary
        # Create new tfidfVectorizer object with old vocabulary
        tfidf_new = TfidfVectorizer(analyzer='word', stop_words = "english", lowercase = True
                                , vocabulary = tfidf_pkl.vocabulary_)

        input_arr=tfidf_new.fit_transform(text)
        input_array=input_arr.toarray()

        #Loading my Classifier pickle file on which the dataset is trained upon
        pickle_in_RF=open('RandomForestClassifier_model.pkl','rb')
        RF_pkl=pickle.load(pickle_in_RF)

        prediction=RF_pkl.predict(input_array)
        print(prediction[0])
        
        if prediction[0]==0:
            pred="Negative"
            #return render_template('index.html',prediction_texts="Negative")
        if prediction[0]==1:
            pred="Positive"
            #return render_template('index.html',prediction_texts="Positive")
        elif prediction[0]==2:
            pred="Neutral"
            #return render_template('index.html',prediction_texts="Neutral")
    #else:
        #return render_template('index.html',prediction_texts="Result not found")

        return render_template('index.html', prediction_text='The predicted sentiment is :{}'.format(pred))


if __name__=="__main__":
    #app.debug = True
    app.run()
