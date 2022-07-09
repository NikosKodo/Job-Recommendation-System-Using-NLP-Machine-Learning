import pandas as pd
import numpy as np
import re
import nltk
pd.set_option('display.max_columns', None)

#εισαγωγή βιβλιοθηκών

df = pd.read_csv("JobsDataset.csv")
df.head()

#άνοιγμα του αρχείου JobsDataset

len(df)

df['Description'][0]

df['clean_Description'] = df['Description'].str.lower()
df['clean_Description'] = df['clean_Description'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_Description'] = df['clean_Description'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_Description']

#Κάνουμε clean την στήλη Description.Αφαιρούμε τα κεφαλαία σύμβολα όπως και κάποια κενά και σημεία στίξης 

df['clean_Description'] = df['clean_Description'].apply(lambda x: nltk.word_tokenize(x))
df['clean_Description']

#Με το nltk.word_tokenize ξεχωρίζουμε την κάθε λέξη

stop_words = nltk.corpus.stopwords.words('english')
Description = []
for sentence in df['clean_Description']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    Description.append(temp)
Description

#Ομαδοποίηση όλων των tokens της καινούργιας στήλης clean_description

df['clean_Description'] = Description

#Ορίζω σαν clean_Description = Description

df['clean_Description'] #Εμφάνιση

df.head()

df['Job Title'] = df['Job Title'].apply(lambda x: x.split(',')[:4])

#Στην στήλη Job Title ξεχωρίζω τις ΄λέξεις μεταξύ τους

df['Job Title'][0] #παράδειγμα εμφάνισης στοιχείου 0

def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp

#Βγάζουμε τα κεφαλαία και τα κενά

df['Job Title'] = [clean(x) for x in df['Job Title']] 

#Χρησιμοποιούμε την συνάρτηση clean που φτιάξαμε προηγουμένος

df['Job Title'][0]

#παράδειγμα εμφάνισης στοιχείου 0

columns = ['clean_Description','Job Title']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
l

#Σύπτυξη clean_Description με Job Title 

df['clean_input'] = l
df = df[['Query', 'clean_input']]
df.head()

#Δήλωση πίνακα l σε clean_input 

#Εμφάνιση Query και clean_input

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_Query = TfidfVectorizer()
X = tfidf_Query.fit_transform(df['clean_input'].values) #fitting and transforming the vector
print(X.shape)

#Με την βιβλιοθήκη TfidfVectorizer

from sklearn.cluster import KMeans

wcss = []
for i in range (1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42, max_iter=600, n_init=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#Φόρτωση της βιβλιοθήκης sklearn.cluster με σκοπό να αναπαραστήσω τον γράφο KMeans. Δημιουργώ μια for για ένα τυχαίο δείγμα.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.lineplot(range(1,15),wcss,marker='o', color = 'red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

#Εμφάνιση του γράφου θέτοντας τις παραπάνω συνιστώσες στις συντετατγμένες x,y

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components =2)

Y_sklearn = sklearn_pca.fit_transform(X.toarray())
kmeans = KMeans(n_clusters = 7, init = 'k-means++', max_iter=600, n_init=1,random_state=42)
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c = prediction, s=50, cmap = 'viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s=300, alpha=0.6);

#Φόρτωση της βιβλιοθήκης sklearn.decomposition με σκοπό να αναπαραστήσω τον γράφο PCA. 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])

#Με την βιβλιοθήκη TfidfVectorizer και χρησιμοποιώντας CountVectorizer μετατρέπουμε τα στοιχεία σε δείκτες και υλοποιούμε την matrix μέτρηση  

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(features, features)
print(cosine_sim)

#Με την χρήση του cosine_similarity υπολογίζουμε μεταξύ των δεικτών του παραπάνω πίνακα 

index = pd.Series(df['Query'])
index.head()

#εμφάνιση σε μορφ΄η στήλης των στοιχείων του Query

def recommend_jobs(title):
    jobs = []
    idx = index[index == title].index[0]
    # print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    # print(top10)
    
    for i in top10:
        jobs.append(df['Query'][i])
    return jobs

#Φτιάχνουμε την συνάρτηση recommend_jobs η οποία δέχεται τις μετρήσεις του cosine_similarity και τυπώνει τις 10 πιο σχετικές αναζητήσεις 

recommend_jobs('Data Analyst') #Εδώ βάζουμε την κατηγορία που ανήκει η θέση εργασίας (Query) και μας β΄γαζει σαν αποτέλεσμα τις 10 πιο σχετικές αναζητήσεις 

pd.Series(cosine_sim[3]).sort_values(ascending=False)

#Τυπώνουμε τα αποτελέσματα του cosine_similarity σε σειρά απο το πιο σχετικό έως πιο αδιάφορο 