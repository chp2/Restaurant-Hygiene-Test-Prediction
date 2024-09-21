from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

nltk.download('punkt')

cuisine_list = ['Salvadoran', 'Live/Raw Food', 'Shanghainese', 'Cuban', 'Turkish', 'Senegalese', 'Gastropubs', 'Dim Sum', 'Cantonese', 'Italian', 'Fast Food', 'Hot Pot', 'Brazilian', 'Halal', 'Szechuan', 'Egyptian', 'Tapas/Small Plates', 'Diners', 'Kosher', 'Burgers', 'Venezuelan', 'Irish', 'Spanish', 'Haitian', 'Chinese', 'Creperies', 'Mexican', 'Delis', 'Buffets', 'Fish & Chips', 'Trinidadian', 'Cheesesteaks', 'Colombian', 'Thai', 'Barbeque', 'Sandwiches', 'Soup', 'Hawaiian', 'Asian Fusion', 'Pakistani', 'Malaysian', 'Sushi Bars', 'Vegetarian', 'Fondue', 'Food Stands', 'Indian', 'Southern', 'German', 'Cambodian', 'Greek', 'Scandinavian', 'British', 'Mediterranean', 'Steakhouses', 'Laotian', 'Comfort Food', 'Hot Dogs', 'Basque', 'Vegan', 'Modern European', 'Australian', 'Russian', 'Puerto Rican', 'Filipino', 'Soul Food', 'Food Court', 'Middle Eastern', 'Seafood', 'Himalayan/Nepalese', 'Moroccan', 'Caribbean', 'Salad', 'Lebanese', 'American (New)', 'Gluten-Free', 'Vietnamese', 'Taiwanese', 'Polish', 'Breakfast & Brunch', 'Cafes', 'Indonesian', 'Cajun/Creole', 'Ethiopian', 'African', 'Tapas Bars', 'Persian/Iranian', 'Mongolian', 'Latin American', 'Pizza', 'American (Traditional)', 'Scottish', 'Japanese', 'Tex-Mex', 'Chicken Wings', 'Belgian', 'Korean', 'Afghan', 'French']
zipcode_list = ['98101', '98102', '98103', '98104', '98105', '98106', '98107', '98108', '98109', '98112', '98115', '98116', '98117', '98118', '98119', '98121', '98122', '98125', '98126', '98133', '98134', '98136', '98144', '98146', '98166', '98168', '98177', '98178', '98188', '98199']

# Add your custom stopwords here
custom_stopwords = {'you', 'they', 'in', '$', '#', ';', '&', '.', 'i', '..', '...', ',','(',')', '-', '?', '!', ':', '*', "'s", "'ll", "n't", '``', "''", "'d"}
stopwords = STOPWORDS.union(custom_stopwords)
stemmer = PorterStemmer()

def splitting (result_text):
    print('result text: ', result_text)
    pattern = re.compile(r"Restaurant: (?P<restaurant_name>.*?), Zipcode: (?P<zipcode>\d{5}), Rating: (?P<rating>\d+\.\d+), Cuisines: (?P<cuisines>.*?), Review: (?P<review>.*)")
    match = pattern.search(result_text)
    if match:
        restaurant = match.group('restaurant_name')
        zipcode = match.group('zipcode')
        rating = match.group('rating')
        cuisines = match.group('cuisines').split(', ')
        review = match.group('review')
        return restaurant, zipcode, rating, cuisines, review
    else:
        print("No match found")
        return -1, -1, -1, -1, -1

def preprocess_text(text):
    text = text.lower()
    token_list = word_tokenize(text)
    filtered_token_list = [word for word in token_list if word not in stopwords]
    filtered_tokens = [stemmer.stem(word) for word in filtered_token_list]
    return filtered_tokens

def feature_normalizing(zipcode, cuisines, rating):
    zipcode_label = []
    for zipcode_category in zipcode_list:
        if zipcode_category == zipcode:
            zipcode_label.append(1)
        else:
            zipcode_label.append(0)

    category_label = []
    for cuisine in cuisine_list:
        if cuisine in cuisines:
            category_label.append(1)
        else:
            category_label.append(0)

    normalized_rating = float(rating)/5

    return zipcode_label, normalized_rating, category_label

print("Beginning Flask")
#Flask construction
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/hygiene_test', methods=['POST'])
def hygiene_test():
    data = request.json
    result_text = data.get('resultText')
    restaurant, zipcode, rating, cuisines, review = splitting(str(result_text))
    if restaurant == -1:
        return jsonify({"message": "Received", "resultText": "Failed!"})
    zipcode_label, normalized_rating, category_label = feature_normalizing(zipcode, cuisines, rating)
    print(zipcode_label)
    print(normalized_rating)
    print(category_label)
    print(review)
    tokenized_text = preprocess_text(review)
    doc = id2word.doc2bow(tokenized_text)
    doc_topics = lda_model.get_document_topics(doc, minimum_probability=0)
    topic_probability_vector = [topic_prob for topic_id, topic_prob in doc_topics]
    print(topic_probability_vector)
    feature_matrix = [topic_probability_vector + zipcode_label + [0.01] + [normalized_rating] + category_label]
    X_test = pd.DataFrame(feature_matrix, columns=column_names)
    print(X_test)
    X_test = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test)
    print(y_pred)
    if y_pred[0] == 0:
        processed_text = restaurant + " Passed!"
    else:
        processed_text = restaurant + " Failed!"    
    return jsonify({"message": "Received", "resultText": processed_text})

if __name__ == '__main__':
    #construct LDA model
    print("beginning constructing LDA model")
    review_file = 'hygiene.dat'
    f = open(review_file, 'r', encoding='utf-8')
    review_list = f.readlines()

    # Tokenize the review text, remove stopwords, and take stems
    print("tokenizing")
    filtered_tokens = []
    for review in review_list:
        stemmed_token_list = preprocess_text(review)
        filtered_tokens.append(stemmed_token_list)
    
    print("create dictionary and corpus")
    # Create a dictionary and corpus
    id2word = corpora.Dictionary(filtered_tokens)
    corpus = [id2word.doc2bow(text) for text in filtered_tokens]

    print("training")
    # Train the LDA model
    num_topics = 10
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, update_every=1, passes=10, alpha='auto', per_word_topics=True)

    #Construct SVM & train
    print("Beginning onstructing SVM")
    column_names = ['c'+str(name) for name in range(10+30+1+1+98)] # Define column names manually
    # Load the dataset from a CSV file
    X_train = pd.read_csv('X_train_LDA.csv', header=None, names=column_names)
    y_train = pd.read_csv('Y_train.csv', header=None, names=['target'])
    y_train = y_train.values.ravel()
    print("standardizing")
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print("Initialize the SVM model")
    # Initialize the SVM model
    svm_model = SVC(kernel='linear')

    # Train the model on the training data
    print("Beginning training svm")
    svm_model.fit(X_train, y_train)

    app.run(debug=True, use_reloader=False)