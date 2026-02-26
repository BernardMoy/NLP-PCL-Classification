import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

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

#================================================================================
"""
Duplicates, special characters, 
Keyword distribution, 
label distribution (YES OR NO) within each keyword and overall, 
text length distribution 
"""

# Show NA values 
print("NA values: ")
print(df.isna().sum()) 
print("="*50)

# Drop NA values for now 
df = df.dropna()

# Check for special characters 
print("Special characters:")
special_characters = ['&amp;', '&lt;', '&gt;', '<h>', '\n', '\t']
for char in special_characters: 
    rows = df[df["text"].str.contains(char, regex=False, na=False)]
    print(f"Rows containing {char}:")
    print(rows if rows.shape[0] > 0 else f"No rows.")
print("="*50)

# Show the distribution of keywords 
print("Distribution of keywords:")
keyword_counter = df["keyword"].value_counts() 
print(keyword_counter)
print("="*50) 

# Show the distribution of sentence length 
print("Distribution of number of words:")
sentence_length = df["text"].apply(lambda x: len(x.split()))
plt.figure()
plt.xlim(0, 300) 
plt.hist(sentence_length, bins=100)
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.title("Distribution of Number of words")
plt.show()

print("Min, max, mean sentence length:")
print(min(sentence_length), max(sentence_length), np.mean(sentence_length))
print("="*50) 

# Check for duplicates of sentences 
print("Duplicates of sentences:")
duplicates = df[df["text"].duplicated()] 
print(duplicates if duplicates.shape[0] > 0 else "No duplicates.")
print("="*50) 


# Label distribution, for each keyword
print("Label distribution for each keyword:")
labels = df.groupby("keyword")["label"].value_counts()
print(labels) 
labels_yesno = (df["label"] > 1).groupby(df["keyword"]).value_counts()
print(labels_yesno)
print("="*50) 

# Label distribution overall 
print("Label distribution overall:") 
labels = df["label"].value_counts() 
print(labels)
labels_yesno = df["label"].apply(lambda x: x > 1 ).value_counts()
print(labels_yesno)
print("="*50) 

#================================================================================
"""
Cooccurrence matrix of keywords, 
n-gram, 
TSNE
"""