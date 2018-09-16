import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk

if __name__ == '__main__':
    print("Read the data onto pandas dataframes")
    train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                             header=0, delimiter="\t", quoting=3)
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'),
                            header=0, delimiter="\t", quoting=3)
  
    print(f"The first review is {train_data['review'][0]}")

    print("Cleaning and parsing the training set movie reviews\n")
    clean_train_reviews = []

    for review in train_data['review']:
        cleaned_review = " ".join(KaggleWord2VecUtility.review_to_wordlist(review, True))
        clean_train_reviews.append(cleaned_review)

    print(clean_train_reviews[0])

    print("Creating Bag of Words Feature Representation: \n")
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

    print("Creating Random Forest Classifier and Training it: \n")
    random_forest = RandomForestClassifier(n_estimators=200).fit(train_data_features, train_data['sentiment'])

    print("Cleaning and parsing the test set movie reviews: \n")
    clean_test_reviews = []

    for review in test_data['review']:
        cleaned_review = " ".join(KaggleWord2VecUtility.review_to_wordlist(review, True))
        clean_test_reviews.append(cleaned_review)

    print(clean_test_reviews[0])

    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

    print("Predicting test data labels:\n")
    result = random_forest.predict(test_data_features)
    output = pd.DataFrame( data={"id": test_data['id'], "sentiment": result} )
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'bag_of_words_inference.csv'),
                  index=False, quoting=3)

    print("Wrote Inference Data")    
