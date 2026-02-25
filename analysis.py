import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

# Read the pcl.tsv dataset 
df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t')

# Show the distribution of keywords 
keyword_counter = df["keyword"].value_counts() 

print(df.head())
print(keyword_counter)