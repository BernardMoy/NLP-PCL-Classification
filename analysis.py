import nltk
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk import ngrams
from nltk.corpus import stopwords
import string

# Define constants 
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
print(f"{len(STOPWORDS)} stopwords loaded")
print("="*50)

SPECIAL_CHARACTERS = ['&amp;', '&lt;', '&gt;', '<h>', '\n', '\t']

# Read the pcl.tsv dataset 
df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t')
df_test = pd.read_csv('data/task4_test.tsv', sep='\t')

"""
The two dontpatronize me tsv files have been modified. 
Check this during submission 
"""

# Obtain train and val labels 
train_labels = pd.read_csv('data/train_semeval_parids-labels.csv')["par_id"]
val_labels = pd.read_csv('data/dev_semeval_parids-labels.csv')["par_id"]

# Obtain train and validation data a
df_train = df[df["par_id"].isin(train_labels)]
df_val = df[df["par_id"].isin(val_labels)]

# print(df.head())
# print(df_train.head()) 
# print(df_test.head())

# Show NA values 
print("NA values: ")
print(df.isna().sum()) 
print("="*50)

# Drop NA values for now 
df = df.dropna()

# Utility function that returns whether a n-gram contains stopwords 
def contains_stopwords(gram): 
    if isinstance(gram, str): 
        s = set(gram.split())
        return len(s.intersection(STOPWORDS)) != 0 
    elif isinstance(gram, tuple):  
        return len(set(gram).intersection(STOPWORDS)) != 0 

#================================================================================
def univariate_analysis(): 
    # Show the distribution of keywords 
    print("Distribution of keywords:")
    keyword_counter = df["keyword"].value_counts() 
    print(keyword_counter)
    print("="*50) 


    # Show the distribution of sentence length 
    print("Distribution of sentence length:")
    sentence_length = df["text"].apply(lambda x: len(x.split()))
    plt.figure()
    plt.xlim(0, 200) 
    plt.hist(sentence_length, bins=100)
    plt.xlabel("Sentence length")
    plt.ylabel("Frequency")
    plt.title("Distribution of sentence length")
    plt.show()

    print("Min, max, mean sentence length:")
    print(min(sentence_length), max(sentence_length), np.mean(sentence_length), np.median(sentence_length))
    print("="*50) 

    print("95 percentile sentence length: ")
    print(np.percentile(sentence_length, [2.5,97.5]))

    # Check for duplicates of sentences 
    print("Duplicates of sentences:")
    duplicates = df[df["text"].duplicated()] 
    print(duplicates if duplicates.shape[0] > 0 else "No duplicates.")
    print("="*50) 


    # # Label distribution overall 
    # print("Label distribution overall:") 
    # labels = df["label"].value_counts() 
    # print(labels)
    # labels_yesno = df["label"].apply(lambda x: x > 1 ).value_counts()
    # print(labels_yesno)
    # print("="*50) 


    """
    Number of unique words, filler words, special characters, n-gram (2,3)
    """
    # Show the number of unique sentence words 
    print("Number of unique words:")
    unique = df["text"].str.split().explode().nunique()
    print(unique)
    print("="*50)

    # Check for special characters 
    print("Special characters:")
    for char in SPECIAL_CHARACTERS: 
        rows = df[df["text"].str.contains(char, regex=False, na=False)]
        print(f"Rows containing {char}:")
        print(rows if rows.shape[0] > 0 else f"No rows.")
    print("="*50)


    """
    To create cleaned text columns: 
    - Remove special characters
    - Remove 's
    - Remove punctuations
    - Lowercase 
    """
    # Create text column without stop words 
    text_cleaned = df["text"].apply(
        lambda x: " ".join([w for w in x.lower().split() if w not in string.punctuation and w != "'s" and w not in SPECIAL_CHARACTERS])
    )

    # print(text_cleaned.head())

    # Check for the most frequent n-grams  

    # Remove n-grams with stopwords, do this after obtaining the n-grams to make sure
    # all n-grams existed in the original document 
    print(f"Top 20 most frequent bigrams:")
    bigrams = [x for x in ngrams(text_cleaned.str.split().explode(), 2) if not contains_stopwords(x)] 
    bi_gram_counts = Counter(bigrams) 
    print(bi_gram_counts.most_common(20)) 

    print(f"Top 20 most frequent trigrams:")
    trigrams = [x for x in ngrams(text_cleaned.str.split().explode(), 3) if not contains_stopwords(x)] 
    tri_gram_counts = Counter(trigrams)
    print(tri_gram_counts.most_common(20)) 
    print("="*50)

    # TF-IDF if have time 

#================================================================================

def bivariate_analysis(): 
    # Label distribution, for each keyword
    print("Label distribution for each keyword:")
    labels = df.groupby("keyword")["label"].value_counts()
    print(labels) 
    labels_yesno = (df["label"] > 1).groupby(df["keyword"]).value_counts()
    print(labels_yesno)

    # plot 
    keywords = df["keyword"].unique() 
    unique_kw = len(keywords)
    colors = {
        0: '#8fffad',
        1: '#b8ffcb',
        2: '#ff9ecb',
        3: '#ff7dba',
        4: '#ff4fa1'
    }
    fig, ax = plt.subplots(1, unique_kw, figsize = (5*unique_kw, 5))
    for a, k in zip(ax, keywords): 
        data = df[df["keyword"] == k]["label"].value_counts()  # get counter for label for this keyword
        labels = [str(label) for label in data.index]   # Extract labels 0-5 from the value counts dict
        colors = [colors[label] for label in data.index]  # Set green for not PCL, red for PCL
        a.pie(data.values, labels = labels, colors = colors) 
        a.set_title(k)
    plt.show() 

    print("="*50) 


    # Find out which uni-gram / bi-gram is most associated with the label 
    vectorizer = CountVectorizer(
        ngram_range=(2,3),
        stop_words=[], 
        min_df=5  # Require each n-gram to appear in at least 5 documents, otherwise the result will be full of one appearance of it with high score
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["label"] > 1  # Use binary classification here 
    print(X.shape, y.shape)

    names = vectorizer.get_feature_names_out()

    model = LogisticRegression(class_weight='balanced') 
    model.fit(X,y) 
    coefs = np.round(model.coef_[0], 2)

    # Filter out n-grams that contain stop words. 
    coefs_and_names = zip(coefs, names)
    coefs_and_names = [(c, n) for c,n in coefs_and_names if not contains_stopwords(n)]

    # Find the coefficients that is most likely and least likely PCL 
    most_likely = sorted(coefs_and_names, reverse=True)[:20]
    least_likely = sorted(coefs_and_names)[:20]

    print("bi-grams or tri-grams most likely lead to PCL:")
    print(most_likely) 

    print("bi-grams or tri-grams least likely lead to PCL:")
    print(least_likely)


# Show results 
univariate_analysis() 
bivariate_analysis() 