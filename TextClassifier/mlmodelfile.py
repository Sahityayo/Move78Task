import os
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from django.conf import settings

# Regular expressions for removing unwanted characters from dataset

Space_Replacement_Re = re.compile('[/(){}\[\]\|@,;]')
Special_Characters_Re = re.compile('[^0-9a-z #+_]')
StopWords = set(stopwords.words('english'))
CSS_custom_regex= '(?s)(.*?;)'
CSS_Replacement_Re = re.compile(CSS_custom_regex)

# Function for cleaning the data
def clean_text(text):
    text = re.sub(CSS_Replacement_Re, "", text)
    text = BeautifulSoup(text, "lxml").text  # For HTML decoding
    text = text.lower()
    text = Space_Replacement_Re.sub(' ', text)
    text = Special_Characters_Re.sub('', text)
    text = ' '.join(word for word in text.split() if word not in StopWords)  # Remove the stopwords
    text = text.strip()
    return text


def machine_learning(input):
    # Path of the dataset
    dataset_path = os.path.join(settings.MLDATA_ROOT, 'mail_data.xlsx')

    # Converting dataset into DataFrame
    df = pd.read_excel(dataset_path)
    if df.empty == True:
        return

    df = df[pd.notnull(df['Category'])]

    # Clean the data
    df['Mail'] = df['Mail'].apply(clean_text)

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(df['Mail'], df['Category'], random_state=42, test_size=0.3)

    # Feature Extraction
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Applying the classifier
    # clf = SVC(kernel='linear',probability = True).fit(X_train_tfidf, y_train)

    # Pickling the model to avoid running again
    # save_classifier = open('model.pickle',"wb")
    # pickle.dump(clf,save_classifier)
    # save_classifier.close()

    # Using pickle file for prediction
    pickle_path = os.path.join(settings.MLDATA_ROOT, 'model.pickle')
    try:
        classifier_file = open(pickle_path, 'rb')
    except IOError:
        return "Could not open file"
    classifier = pickle.load(classifier_file)
    y_pred = classifier.predict(count_vect.transform(X_test))

    # Accuracy of the classifier
    # print('Accuracy %s' % accuracy_score(y_pred, y_test))

    # Predict the category for a given input email body
    input_cv = count_vect.transform([input])
    category = classifier.predict(input_cv)

    # Calculating the confidence
    conf_df = pd.DataFrame(classifier.predict_proba(input_cv), columns=classifier.classes_)

    # Return the predicted result
    resultjson = {"intent" : category[0], "confidence":conf_df.at[0, category[0]]}
    return resultjson