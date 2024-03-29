
# coding: utf-8

# In[3]:


import re,string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import punkt
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

  
from string import punctuation
from collections import Counter
 
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from afinn import Afinn
from wordcloud import WordCloud, STOPWORDS




# In[62]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

import matplotlib.pyplot as plt


# In[52]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup


# In[27]:


porter = PorterStemmer()
wnl = WordNetLemmatizer() 


# In[28]:


stop_words = ["restaurant", "people", "food", "place", "another", "asked", "us", "know","will","food","right","said",
             "got","explained", "phoenix"] + list(STOPWORDS)

stopwords =set(stop_words) 


# In[6]:


STEMMING = True  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 2  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec

def pos_tag_text(tokens):
    #Convert Penn treebank tag to wordnet because lemmatize uses wordnet tags
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    
    #tokens=word_tokenize(text)
    tagged_text = pos_tag(tokens,tagset='universal')
    tagged_tokens = [(word, penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    return tagged_tokens

def lemmatize_pos_text(tokens):
    pos_tagged_text=pos_tag_text(tokens)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word,pos_tag in pos_tagged_text]
#    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_tokens

###############################################################################
### Function to process documents
###############################################################################
def clean_doc(doc): 
    # split document into individual words
    tokens=doc.split()
    
    tokens = lemmatize_pos_text(tokens)
    
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # # filter out short tokens
    tokens = [word for word in tokens if len(word) > 3]
    # #lowercase all words
    tokens = [word.lower() for word in tokens]
    # # filter out stop words
    custom_stop_words=['people','american','america','nation','congress']
    english_stop_words = (stopwords.words('english'))
    stopword_list=english_stop_words+custom_stop_words
    tokens = [w for w in tokens if not w in stopword_list]         
    # # word stemming Commented
    if STEMMING:
        ps=PorterStemmer()
        tokens=[ps.stem(word) for word in tokens]
    return tokens


# In[7]:


def get_keywords(tokens, num):
    return Counter(tokens).most_common(num)


# In[10]:


def build_text(text):
    review_text = []
    for index, row in text.iterrows():
        try:
            data=row['text'].strip().replace("'", "")
            #data = strip_tags(data)
            soup = BeautifulSoup(data , "lxml")
            data = soup.get_text()
            data = data.encode('ascii', 'ignore').decode('ascii')
            document = tokenizer(data)
           # document = clean_doc(data)
            top_15 = get_keywords(document, 500)
           # print(top_5)
            unzipped = next(zip(*top_15))
            kw= list(unzipped)
            kw=",".join(str(x) for x in kw)
            review_text.append((kw, row['business_id'], row['business_star_rating'],row['user_star_rating'],row['date'],row['categories']))
        except Exception as e:
            print(e)
            #print data
            #break
            pass
        #break
    review_text_df = pd.DataFrame(review_text, columns=['keywords', 'business_id', 'business_star_rating','user_star_rating','date','categories'])
    return review_text_df


# In[6]:


business = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/business.csv")
#business.head(100)
business.count() #3244 cats, 3247 rest
#business.shape#3247 rows, 22 columns
#business.nunique()#68 cat, review_count 459
#business.isnull().sum()#3 null cats
#yrv


# In[7]:


business.loc[business[('review_count')].idxmax()]#Pizzeria Bianco, 2035 reviews, italian category


# In[14]:


business.loc[business[('review_count')].idxmin()]#Taco Bell, review count 3,cat Mexican


# In[15]:


business['cat'].value_counts()


# In[16]:


revcount = business.groupby('cat', as_index=False)['review_count'].sum()
#revcount.sort_values['review_count']#mexican -39193, american(new) - 31242,
revcount#mexican -39193, american(new) - 31242,


# In[17]:


ax = sns.barplot(x="cat", y="review_count", data=revcount)


# In[8]:


reviews = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_review.csv")
#reviews.count()
#reviews['text'].head(100)
reviews.nunique()


# In[16]:


reviews.nunique()
reviews.isnull().sum()


# In[17]:


#busreview = pd.merge(business,reviews,on='business_id', how = 'left')##not needed..merge review and business after merging 
##tip and review
#busreview.count()#yelpbusreview.head(15) #294858
#yelpbusreview.nunique()


# In[18]:


#busreview.isnull().sum()#17 cats are null
#busreview.rename(columns={'name':'business_name','stars_x':'business_star_rating','stars_y':'user_star_rating'})


# In[19]:


##combining reviews of users for a same business id by date
#busreview.groupby(['user_id','business_id', 'date'])['text'].apply(lambda x: ','.join(x)).reset_index()


# In[9]:


tips = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_tip.csv")
tips.head()


# In[10]:


business_ids = business.business_id.unique().tolist()
len(business_ids) # 3241


# In[13]:


tip=tips[tips.business_id.isin(business_ids)]
review = reviews[reviews.business_id.isin(business_ids)]
review.count()


# In[22]:


tips.count()


# In[14]:


tip=tips[tips.business_id.isin(business_ids)]
tip.count()


# In[15]:


tipreview = review.append(tip,ignore_index=True)
tipreview.count()


# In[49]:


#tipreview.groupby(['user_id','business_id', 'date'])['text'].apply(lambda x: ','.join(x)).reset_index()
#tipreview.count()
bustipreview = pd.merge(business,tipreview,on='business_id', how = 'left')
bustipreview.count()#367353


# In[53]:


bustipreview = bustipreview.rename(columns={'name':'business_name','stars_x':'business_star_rating','stars_y':'user_star_rating'})
bustipreview = bustipreview.drop(['cool','funny','likes','useful'],axis=1)


# In[27]:


bustipreview.dtypes


# In[29]:


revcount = bustipreview.groupby('cat', as_index=False)['review_count'].sum().reset_index().sort_values('review_count', ascending=False)
#revcount.sort_values['review_count']#mexican -39193, american(new) - 31242,
revcount#mexican -39193, american(new) - 31242,
type(revcount)
revcount.to_csv('C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/reviewcountbycategory.csv',sep=',',index= False)


# In[30]:


revcount


# In[31]:


plt.figure(figsize=(20,10))
ax = sns.barplot(x="cat", y="review_count", data=revcount)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/review_countByCat.png")


# In[32]:


#ax1 = sns.count(x="cat", y="review_count", data=revcount, order=revcount['review_count'].value_counts().index)


# In[33]:


busidmaxtipreview=bustipreview.loc[bustipreview[('review_count')].idxmax()]['business_id']#Pizzeria Bianco, 2035 reviews, italian category
busidmintipreview=bustipreview.loc[bustipreview[('review_count')].idxmin()]['business_id']


# In[34]:


busidmaxtipreview


# In[35]:


bustipreviewmaxdf=bustipreview.loc[bustipreview['business_id']==busidmaxtipreview]
bustipreviewmindf=bustipreview.loc[bustipreview['business_id']==busidmintipreview]


# In[36]:


(bustipreviewmaxdf.groupby(['user_star_rating','date']).size()).plot(color =  'slateblue')
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/maxreviewbus_userratingdistrib.png")


# In[37]:


bustipreview5star = bustipreview.loc[bustipreview['business_star_rating']==5]
bustipreview1star = bustipreview.loc[bustipreview['business_star_rating']==1]
bustipreview2star = bustipreview.loc[bustipreview['business_star_rating']==2]
bustipreview3star = bustipreview.loc[bustipreview['business_star_rating']==3]
bustipreview4star = bustipreview.loc[bustipreview['business_star_rating']==4]


# In[38]:


(bustipreviewmindf.groupby(['user_star_rating','date']).size()).plot(color =  'slateblue')
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/minreviewbus_userratingdistrib.png")


# In[39]:


bustipreview.loc[bustipreview[('review_count')].idxmax()]#Pizzeria Bianco, 2035 reviews, italian category


# In[40]:


businessratinggroup =bustipreview['business_id'].groupby(bustipreview['business_star_rating']).apply(lambda x: ','.join(x)).reset_index()# == busidmaxtipreview]


# In[41]:


users = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_user.csv")
users.count()
users.head(100)
users.nunique()


# In[42]:


bustipreview_df = bustipreview[['business_id','business_star_rating','user_star_rating','date','text','cat','user_id']]
#bustipreview_df = busin[['business_id','text','cat']]


# In[43]:


bustipreview_df


# In[44]:


bustipreview_df.count()


# In[45]:


bustipreview_df.nunique()


# In[46]:


#data = []
#for index, row in yelpbustipreview_df.iterrows():
  #  data.append((row['business_id'], row['stars_x'], row['stars_y'], row['date'],row['text'],row['cat'],row['user_id']))
#yelprev_df = pd.DataFrame(data, columns=['business_id' ,'stars_x','stars_y', 'date', 'text' ,'cat','user_id'])


# In[47]:


#yelpyelprev_df1 = yelprev1[1:200]


# In[48]:


bustipreview_df.groupby(['business_id'])['text'].apply(lambda x: ','.join(x)).reset_index()


# In[49]:


bustipreviewmaxdf.groupby(['text','date'], as_index = False).size()


# In[28]:





# In[29]:


comment_words = ' '
#stopwords =set(STOPWORDS) 

# iterate through the csv file 
for val in bustipreviewmaxdf.text: 
    # typecaste each val to string 
    val = str(val) 

    # split the value 
    tokens = val.split()
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    
    for words in tokens: 
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords=stopwords,
                      min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()


# In[30]:


comment_words = ' '
#stopwords =set(STOPWORDS) 

# iterate through the csv file 
for val in bustipreviewmindf.text: 
    # typecaste each val to string 
    val = str(val) 

    # split the value 
    tokens = val.split()
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    
    for words in tokens: 
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords=stopwords,
                      min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()


# In[53]:


comment_words = ' '
#stopwords =set(STOPWORDS) 

# iterate through the csv file 
for val in bustipreview1star.text: 
    # typecaste each val to string 
    val = str(val) 

    # split the value 
    tokens = val.split()
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    
    for words in tokens: 
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords=stopwords,
                      min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()


# In[54]:


comment_words = ' '
#stopwords =set(STOPWORDS) 

# iterate through the csv file 
for val in bustipreview5star.text: 
    # typecaste each val to string 
    val = str(val) 

    # split the value 
    tokens = val.split()
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    
    for words in tokens: 
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords=stopwords,
                      min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()


# In[55]:


def word_count(text_string):
    '''Calculate the number of words in a string'''
    return len(text_string.split())


# In[56]:


af = Afinn()
sentiment1_scores = [af.score(i) for i in bustipreview1star['text']]
sentiment1_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment1_scores]
bustip1star = pd.DataFrame([list(bustipreview1star['business_id']),list(bustipreview1star['cat']),
                      sentiment1_scores,sentiment1_category]).T
bustip1star.columns = ['business_id','cat','sentiment1_score','sentiment1_category']
bustip1star.groupby(['business_id','cat']).describe()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[57]:


([bustip1star['sentiment1_score'] >=100])
[i for i in bustip1star['sentiment1_score'] if i > 100]


# In[58]:


#bustipreview1star.count()#233
#bustipreview2star.count()#5029
#bustipreview3star.count()#34461
#bustipreview4star.count()#152081
bustipreview5star.count()#2807


# In[59]:


bustipreview2star.count()
af = Afinn()
sentiment2_scores = [af.score(i) for i in bustipreview2star['text']]
sentiment2_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment2_scores]
bustip2star = pd.DataFrame([list(bustipreview2star['business_id']),list(bustipreview2star['cat']),sentiment2_scores,sentiment2_category]).T
bustip2star.columns = ['business_id','cat','sentiment2_score','sentiment2_category']
bustip2star.groupby(['business_id','cat']).describe()
#bustip2star.count()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[60]:


af = Afinn(language='en')

sentiment3_scores = [af.score(i) for i in bustipreview3star['text']]
sentiment3_category = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in sentiment3_scores]
bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','sentiment3_score','sentiment3_category']
bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),list(bustipreview3star['text']),
                            sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','text','sentiment3_score','sentiment3_category']


# In[63]:


bustip3star.groupby(['business_id','cat']).describe()


# In[62]:


af = Afinn(language='en')

sentiment_scores = [af.score(i) for i in bustipreview_df['text']]
sentiment_category = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in sentiment_scores]


# In[ ]:


bustiprev = pd.DataFrame([list(bustipreview_df['business_id']),list(bustipreview_df['cat']),list(bustipreview_df['text']),
                            sentiment_scores,sentiment_category]).T
bustiprev.columns = ['business_id','cat','sentiment_score','sentiment_category']


# In[ ]:


bustiprev['word_count'] = [word_count(i) for i in bustipreview3star['text']]
bustiprev['sentiment_scores_adjusted'] = bustiprev['sentiment_score']/bustiprev['word_count'] *100
bustiprev['sentiment_category_adjusted'] = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in bustiprev['sentiment_scores_adjusted']]
#bustip3star['sentiment3_scores_adjusted'].max()
#bustip3star.groupby(['business_id','cat']).describe()


# In[ ]:


#bustiprev.groupby(['business_id','cat'])['text','sentiment_scores_adjusted','sentiment_category_adjusted'].describe()
bustiprev.groupby(['cat'])['text','sentiment_scores_adjusted','sentiment_category_adjusted'].describe()


# In[64]:


bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),list(bustipreview3star['text']),
                            sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','text','sentiment3_score','sentiment3_category']


# In[65]:


bustip3star['word3_count'] = [word_count(i) for i in bustipreview3star['text']]
bustip3star['sentiment3_scores_adjusted'] = bustip3star['sentiment3_score']/bustip3star['word3_count'] *100
bustip3star['sentiment3_category_adjusted'] = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in bustip3star['sentiment3_scores_adjusted']]
#bustip3star['sentiment3_scores_adjusted'].max()
#bustip3star.groupby(['business_id','cat']).describe()


# In[66]:


(bustip3star.groupby(['cat'])['sentiment3_scores_adjusted','sentiment3_category_adjusted'].describe()).to_csv('C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/3starafinn.csv',sep=',',index= False)
#bustip3star.groupby(['cat'])['sentiment3_category_adjusted'].describe()[]
#bustip3star['cat'].value_counts()


# In[67]:


#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
plt.figure(figsize=(20,10))
sp = sns.stripplot(x='cat', y="sentiment3_scores_adjusted", 
                   hue='cat', data=bustip3star)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
#sp.set_xticklabels(sp.get_xticklabels(), rotation=40)
#sp.set_xticklabels(sp.get_xticklabels(), rotation=40, ha="right")

#bp = sns.boxplot(x='news_category', y="sentiment_score", 
 #                hue='news_category', data=df, palette="Set2", ax=ax2)
#t = f.suptitle('Visualizing Sentiment Score by Categories', fontsize=14)


# In[68]:


plt.figure(figsize=(200,200))
fc = sns.factorplot(x="cat", hue="sentiment3_category_adjusted", 
                    data=bustip3star, kind="count", 
                    palette={"negative": "#FE2020", 
                             "positive": "#BADD07", 
                             "neutral": "#68BFF5"},size=20, aspect=1)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
#fc.set_xticklabels(fc.get_xticklabels(), rotation=40, ha="right")
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/factorplot3starrating.png")


# In[69]:


#[i for i in bustip2star['sentiment2_score'] if i > 90]
max(bustip3star['sentiment3_scores_adjusted'])
#bustip3star[bustip3star['business_id']=='yDKmcWQ_Zycr4ekLV4CqFg']


# In[70]:


plt.figure(figsize=(20,10))
bustip3star['sentiment3_scores_adjusted'] = bustip3star['sentiment3_scores_adjusted'].astype(float)
bp = sns.boxplot(x='cat', y="sentiment3_scores_adjusted", hue='cat', data=bustip3star)


# In[71]:


af = Afinn()
sentiment4_scores = [af.score(i) for i in bustipreview4star['text']]
sentiment4_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
bustip4star = pd.DataFrame([list(bustipreview4star['business_id']),list(bustipreview4star['cat']),sentiment4_scores,sentiment4_category]).T
bustip4star.columns = ['business_id','cat','sentiment4_score','sentiment4_category']
bustip4star.groupby(['business_id','cat']).describe()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[ ]:


af = Afinn()
sentiment5_scores = [af.score(i) for i in bustipreview5star['text']]
sentiment5_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
bustip5star = pd.DataFrame([list(bustipreview5star['business_id']),list(bustipreview4star['cat']),sentiment5_scores,sentiment5_category]).T
bustip5star.columns = ['business_id','cat','sentiment5_score','sentiment5_category']
bustip5star.groupby(['business_id','cat']).describe()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[72]:


build_text(yelprev_df1)


# In[73]:


lemmatizer = WordNetLemmatizer()
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
#def clean_text(text):
 #   text = text.replace("<br />", " ")
  #  text = text.decode("utf-8")
 
  #  return text
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
  #  text = clean_text(text)
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0


# In[ ]:


[swn_polarity(i) for i in yelprev_df1['text']]


# In[ ]:


sentiment_polarity = [swn_polarity(i) for i in yelprev_df1['text']]
yelprev12 = pd.DataFrame([list(yelprev_df1['business_id']),list(yelprev_df1['cat']),sentiment_polarity]).T
yelprev12.columns = ['business_id','cat','sentiment_polarity']
yelprev12.groupby(['business_id','cat']).describe()


# In[74]:


def vaderize(df, textfield):
    '''Compute the Vader polarity scores for a textfield. 
    Returns scores and original dataframe.'''

    analyzer = SentimentIntensityAnalyzer()

    print('Estimating polarity scores for %d cases.' % len(df))
    sentiment = df[textfield].apply(analyzer.polarity_scores)

    # convert to dataframe
    sdf = pd.DataFrame(sentiment.tolist()).add_prefix('vader_')

    # merge dataframes
    df_combined = pd.concat([df, sdf], axis=1)
    return df_combined


# In[75]:


df_vaderized = vaderize(bustipreview3star, 'text')


# In[54]:


bustipreview.head(10)
#bustipreview['business_id'==__aKnGBedQ51_hEc3D9ARw]
#bustipreview.loc[bustipreview['business_id']=='__aKnGBedQ51_hEc3D9ARw'].nunique()


# In[43]:


#bustipreview

#bustipreview['combi_text'] = bustipreview.groupby(['business_id'])['text'].apply(lambda x: ','.join(x)).reset_index()['text']
bustipreview1 = bustipreview.groupby(['business_id'])['text'].apply(lambda x: ','.join(x)).reset_index()
bustipreview1


# In[23]:


#bustipreview.drop('text', axis=1)


# In[44]:


#bustipreview['combi_text'] = bustipreview['combi_text'].str.replace("[^a-zA-Z]", " ")
bustipreview1['text'] = bustipreview['text'].str.replace("[^a-zA-Z]", " ")


# In[40]:


#bustipreview['combi_text'].notnull().value_counts()
#[bustipreview['combi_text'] =='NaN']


# In[45]:


STEMMING = True  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 2  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH = 100  # set vector length for TF-IDF and Doc2Vec

def pos_tag_text(tokens):
    #Convert Penn treebank tag to wordnet because lemmatize uses wordnet tags
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    
    #tokens=word_tokenize(text)
    tagged_text = pos_tag(tokens,tagset='universal')
    tagged_tokens = [(word, penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    return tagged_tokens

def lemmatize_pos_text(tokens):
    pos_tagged_text=pos_tag_text(tokens)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word,pos_tag in pos_tagged_text]
#    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_tokens

###############################################################################
### Function to process documents
###############################################################################
def clean_text(text): 
    # split document into individual words
    tokens=text.split()
    
    tokens = lemmatize_pos_text(tokens)
    
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
   # tokens = [word for word in tokens if word.isalpha()]
    # # filter out short tokens
    tokens = [word for word in tokens if len(word) > 3]
    # #lowercase all words
    tokens = [word.lower() for word in tokens]
    # # filter out stop words
    
    #stopword_list=english_stop_words+custom_stop_words
    tokens = [w for w in tokens if not w in stopwords]         
    # # word stemming Commented
    if STEMMING:
        ps=PorterStemmer()
        tokens=[ps.stem(word) for word in tokens]
    return tokens


# In[46]:


tidy_text = []
for i in bustipreview1['text']:
    text=clean_text(i)
    tidy_text.append(text)


# In[68]:


tidy_text


# In[57]:


for i in range(len(tidy_text)):
    tidy_text[i] = ' '.join(tidy_text[i])    
bustipreview1['tidy_text'] = tidy_text


# In[66]:


#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in Word2Vec), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)
 
###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple-word tokens within the TFIDF matrix
#Call Tfidf Vectorizer
print('\nWorking on TF-IDF vectorization')
Tfidf=TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
	max_features = VECTOR_LENGTH)


# In[67]:


TFIDF_matrix=Tfidf.fit_transform(bustipreview1['tidy_text'])     


# In[ ]:




#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.


#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), 
	columns = Tfidf.get_feature_names(), 
	index = labels)

matrix.to_csv('tfidf-matrix.csv')
print('\nTF-IDF vectorization complete, matrix saved to tfidf-matrix.csv')

###############################################################################
### Explore TFIDF Values
###############################################################################
average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF = pd.DataFrame(average_TFIDF,
	index = [0]).transpose()

average_TFIDF_DF.columns=['TFIDF']

#calculate Q1 and Q3 range
Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

#words that exceed the Q3+IQR*1.5
outlier_list = average_TFIDF_DF[average_TFIDF_DF['TFIDF'] >= outlier]

#can export matrix to csv and explore further if necessary
enum1 = enumerate(final_processed_text)


# In[65]:


bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000) 
bow = bow_vectorizer.fit_transform(bustipreview1['tidy_text']) 
bow.shape


# In[56]:


labels=[]
for i in range(0,len(bustipreview1)):
    temp_text=bustipreview1['business_id'].iloc[i]
    labels.append(temp_text)
#len(result)


#for pres in data['president']:
 #   if
#for pres in data['president']:
#    if(len(result)==0):
#        result.append([pres])
#    else:
#        for i in range(0,len(result)):
#            score=SequenceMatcher(None,pres,result[i][0]).ratio()
#            if(score == 1):
#                if(i==len(result)-1):
#                    result.append([pres])
#            else:
#                if(score != 1):
#                    result[i].append(pres)

#result[0]
    
#labels = sorted(labels)
# for key,group in groupby(labels, lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1]):
#    result.append(list(group))
#create empty list to store text documents
text_body=[]

#for loop which appends the text to the text_body list
for i in range(0,len(bustipreview1)):
    temp_text=bustipreview1['text'].iloc[i]
    text_body.append(temp_text)

    
#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_text(i)
    processed_text.append(text)

#Note: the processed_text is the PROCESSED list of documents read directly from
#the csv.  Note the list of words is separated by commas.

#stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)
    
#the following is an example of what the processed text looks like.  
print('\nExample of what one parsed documnet looks like:\n')
print(final_processed_text[0])
    


# In[ ]:



#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in Word2Vec), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)
 
###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple-word tokens within the TFIDF matrix
#Call Tfidf Vectorizer
print('\nWorking on TF-IDF vectorization')
Tfidf=TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
	max_features = VECTOR_LENGTH)

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), 
	columns = Tfidf.get_feature_names(), 
	index = labels)

matrix.to_csv('tfidf-matrix.csv')
print('\nTF-IDF vectorization complete, matrix saved to tfidf-matrix.csv')

###############################################################################
### Explore TFIDF Values
###############################################################################
average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF = pd.DataFrame(average_TFIDF,
	index = [0]).transpose()

average_TFIDF_DF.columns=['TFIDF']

#calculate Q1 and Q3 range
Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

#words that exceed the Q3+IQR*1.5
outlier_list = average_TFIDF_DF[average_TFIDF_DF['TFIDF'] >= outlier]

#can export matrix to csv and explore further if necessary
enum1 = enumerate(final_processed_text)
###############################################################################
### Doc2Vec
###############################################################################
print("\nWorking on Doc2Vec vectorization")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_text)]
model = Doc2Vec(documents, vector_size = VECTOR_LENGTH, window = 2, 
	min_count = 1, workers = 4)

doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_text)):
    vector=pd.DataFrame(model.infer_vector(processed_text[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index()

doc_titles={'title': labels}
t=pd.DataFrame(doc_titles)

doc2vec_df=pd.concat([doc2vec_df,t], axis=1)

doc2vec_df=doc2vec_df.drop('index', axis=1)
doc2vec_df=doc2vec_df.set_index('title')

doc2vec_df.to_csv('doc2vec-matrix.csv')
print('\nDoc2Vec vectorization complete, matrix saved to doc2vec-matrix.csv')

dictionary = gensim.corpora.Dictionary(processed_text)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]

tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
    
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
lsi_model_tfidf = gensim.models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=10)
for idx, topic in lsi_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

