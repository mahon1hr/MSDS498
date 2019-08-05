
# coding: utf-8

# In[218]:


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
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from afinn import Afinn
from wordcloud import WordCloud, STOPWORDS

from bs4 import BeautifulSoup


# In[2]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag


# In[470]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[3]:


porter = PorterStemmer()
wnl = WordNetLemmatizer() 
stop = stopwords.words('english')
stop.append("new")
stop.append("like")
stop.append("u")
stop.append("it'")
stop.append("'s")
stop.append("n't")
stop.append('mr.')
stop = set(stop)


# In[4]:


def tokenizer(text):
 
    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
 
    tokens = []
    for token_by_sent in tokens_:
        tokens += token_by_sent
 
    tokens = list(filter(lambda t: t.lower() not in stop, tokens))
    tokens = list(filter(lambda t: t not in punctuation, tokens))
    tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', u'\u2014', u'\u2026', u'\u2013'], tokens))
     
    filtered_tokens = []
    for token in tokens:
        token = wnl.lemmatize(token)
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
 
    filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))
 
    return filtered_tokens


# In[5]:


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


# In[6]:


def get_keywords(tokens, num):
    return Counter(tokens).most_common(num)


# In[7]:


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


# In[61]:


business = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/business.csv")
#business.head(100)
business.count() #3244 cats, 3247 rest
#business.shape#3247 rows, 22 columns
#business.nunique()#68 cat, review_count 459
#business.isnull().sum()#3 null cats
#yrv


# In[9]:


business.loc[business[('review_count')].idxmax()]#Pizzeria Bianco, 2035 reviews, italian category


# In[10]:


business.loc[business[('review_count')].idxmin()]#Taco Bell, review count 3,cat Mexican


# In[62]:


business['cat'].value_counts()


# In[63]:


revcount = business.groupby('cat', as_index=False)['review_count'].sum()
#revcount.sort_values['review_count']#mexican -39193, american(new) - 31242,
revcount#mexican -39193, american(new) - 31242,


# In[64]:


ax = sns.barplot(x="cat", y="review_count", data=revcount)


# In[14]:


reviews = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_review.csv")
#reviews.count()
#reviews['text'].head(100)
reviews.nunique()


# In[15]:


reviews.nunique()
reviews.isnull().sum()


# In[16]:


#busreview = pd.merge(business,reviews,on='business_id', how = 'left')##not needed..merge review and business after merging 
##tip and review
#busreview.count()#yelpbusreview.head(15) #294858
#yelpbusreview.nunique()


# In[17]:


#busreview.isnull().sum()#17 cats are null
#busreview.rename(columns={'name':'business_name','stars_x':'business_star_rating','stars_y':'user_star_rating'})


# In[48]:


##combining reviews of users for a same business id by date
#busreview.groupby(['user_id','business_id', 'date'])['text'].apply(lambda x: ','.join(x)).reset_index()


# In[20]:


tips = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_tip.csv")
tips.head()


# In[22]:


business_ids = business.business_id.unique().tolist()
len(business_ids) # 3241


# In[27]:


tip=tips[tips.business_id.isin(business_ids)]
review = reviews[reviews.business_id.isin(business_ids)]
review.count()


# In[221]:


tips.count()


# In[36]:


tip=tips[tips.business_id.isin(business_ids)]
tip.count()


# In[35]:


tipreview = review.append(tip,ignore_index=True)
tipreview.count()


# In[65]:


#tipreview.groupby(['user_id','business_id', 'date'])['text'].apply(lambda x: ','.join(x)).reset_index()
#tipreview.count()
bustipreview = pd.merge(business,tipreview,on='business_id', how = 'left')
bustipreview.count()#367353


# In[67]:


bustipreview = bustipreview.rename(columns={'name':'business_name','stars_x':'business_star_rating','stars_y':'user_star_rating'})
bustipreview.drop(['cool','funny','likes','useful'],axis=1)


# In[68]:


bustipreview.dtypes


# In[259]:


revcount = bustipreview.groupby('cat', as_index=False)['review_count'].sum().reset_index().sort_values('review_count', ascending=False)
#revcount.sort_values['review_count']#mexican -39193, american(new) - 31242,
revcount#mexican -39193, american(new) - 31242,
type(revcount)
revcount.to_csv('C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/reviewcountbycategory.csv',sep=',',index= False)


# In[260]:


revcount


# In[261]:


plt.figure(figsize=(20,10))
ax = sns.barplot(x="cat", y="review_count", data=revcount)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/review_countByCat.png")


# In[239]:


ax1 = sns.count(x="cat", y="review_count", data=revcount, order=revcount['review_count'].value_counts().index)


# In[173]:


busidmaxtipreview=bustipreview.loc[bustipreview[('review_count')].idxmax()]['business_id']#Pizzeria Bianco, 2035 reviews, italian category
busidmintipreview=bustipreview.loc[bustipreview[('review_count')].idxmin()]['business_id']


# In[161]:


busidmaxtipreview


# In[174]:


bustipreviewmaxdf=bustipreview.loc[bustipreview['business_id']==busidmaxtipreview]
bustipreviewmindf=bustipreview.loc[bustipreview['business_id']==busidmintipreview]


# In[255]:


(bustipreviewmaxdf.groupby(['user_star_rating','date']).size()).plot(color =  'slateblue')
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/maxreviewbus_userratingdistrib.png")


# In[271]:


bustipreview5star = bustipreview.loc[bustipreview['business_star_rating']==5]
bustipreview1star = bustipreview.loc[bustipreview['business_star_rating']==1]
bustipreview2star = bustipreview.loc[bustipreview['business_star_rating']==2]
bustipreview3star = bustipreview.loc[bustipreview['business_star_rating']==3]
bustipreview4star = bustipreview.loc[bustipreview['business_star_rating']==4]


# In[256]:


(bustipreviewmindf.groupby(['user_star_rating','date']).size()).plot(color =  'slateblue')
plt.savefig("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/minreviewbus_userratingdistrib.png")


# In[132]:


bustipreview.loc[bustipreview[('review_count')].idxmax()]#Pizzeria Bianco, 2035 reviews, italian category


# In[131]:


businessratinggroup =bustipreview['business_id'].groupby(bustipreview['business_star_rating']).apply(lambda x: ','.join(x)).reset_index()# == busidmaxtipreview]


# In[57]:


#users = pd.read_csv("C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/yelp_user.csv")
users.count()
users.head(100)
users.nunique()


# In[134]:


bustipreview_df = bustipreview[['business_id','business_star_rating','user_star_rating','date','text','cat','user_id']]
#bustipreview_df = busin[['business_id','text','cat']]


# In[135]:


bustipreview_df


# In[137]:


bustipreview_df.count()


# In[140]:


bustipreview_df.nunique()


# In[ ]:


#data = []
#for index, row in yelpbustipreview_df.iterrows():
  #  data.append((row['business_id'], row['stars_x'], row['stars_y'], row['date'],row['text'],row['cat'],row['user_id']))
#yelprev_df = pd.DataFrame(data, columns=['business_id' ,'stars_x','stars_y', 'date', 'text' ,'cat','user_id'])


# In[137]:


#yelpyelprev_df1 = yelprev1[1:200]


# In[151]:


bustipreview_df.groupby(['business_id'])['text'].apply(lambda x: ','.join(x)).reset_index()


# In[195]:


bustipreviewmaxdf.groupby(['text','date'], as_index = False).size()


# In[262]:


stop_words = ["restaurant", "people", "food", "place", "another", "asked", "us", "know","will","food","right","said",
             "got","explained", "phoenix"] + list(STOPWORDS)

stopwords =set(stop_words) 


# In[263]:


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


# In[265]:


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


# In[266]:


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


# In[267]:


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


# In[394]:


def word_count(text_string):
    '''Calculate the number of words in a string'''
    return len(text_string.split())


# In[322]:


af = Afinn()
sentiment1_scores = [af.score(i) for i in bustipreview1star['text']]
sentiment1_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
bustip1star = pd.DataFrame([list(bustipreview1star['business_id']),list(bustipreview1star['cat']),
                      sentiment_scores,sentiment_category]).T
bustip1star.columns = ['business_id','cat','sentiment1_score','sentiment1_category']
bustip1star.groupby(['business_id','cat']).describe()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[314]:


#([bustip1star['sentiment1_score'] >=100])
[i for i in bustip1star['sentiment1_score'] if i > 100]


# In[277]:


#bustipreview1star.count()#233
#bustipreview2star.count()#5029
#bustipreview3star.count()#34461
#bustipreview4star.count()#152081
bustipreview5star.count()#2807


# In[335]:


bustipreview2star.count()
af = Afinn()
sentiment2_scores = [af.score(i) for i in bustipreview2star['text']]
sentiment2_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
bustip2star = pd.DataFrame([list(bustipreview2star['business_id']),list(bustipreview2star['cat']),sentiment2_scores,sentiment2_category]).T
bustip2star.columns = ['business_id','cat','sentiment2_score','sentiment2_category']
bustip2star.groupby(['business_id','cat']).describe()
#bustip2star.count()
#df = pd.DataFrame([list(news_df['news_category']), sentiment_scores, sentiment_category]).T
#df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
#df['sentiment_score'] = df.sentiment_score.astype('float')
#df.groupby(by=['news_category']).describe()


# In[354]:


af = Afinn(language='en')

sentiment3_scores = [af.score(i) for i in bustipreview3star['text']]
sentiment3_category = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in sentiment_scores]
bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),listbustipreview3star['text'],
                            sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','sentiment3_score','sentiment3_category']
bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),list(bustipreview3star['text']),
                            sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','text','sentiment3_score','sentiment3_category']


# In[396]:


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


# In[378]:


bustip3star = pd.DataFrame([list(bustipreview3star['business_id']),list(bustipreview3star['cat']),list(bustipreview3star['text']),
                            sentiment3_scores,sentiment3_category]).T
bustip3star.columns = ['business_id','cat','text','sentiment3_score','sentiment3_category']


# In[391]:


bustip3star['word3_count'] = [word_count(i) for i in bustipreview3star['text']]
bustip3star['sentiment3_scores_adjusted'] = bustip3star['sentiment3_score']/bustip3star['word3_count'] *100
bustip3star['sentiment3_category_adjusted'] = ['positive' if score > 0.0
                          else 'negative' if score < 0.0
                              else 'neutral' 
                                  for score in bustip3star['sentiment3_scores_adjusted']]
#bustip3star['sentiment3_scores_adjusted'].max()
#bustip3star.groupby(['business_id','cat']).describe()


# In[460]:


(bustip3star.groupby(['cat'])['sentiment3_scores_adjusted','sentiment3_category_adjusted'].describe()).to_csv('C:/Users/Ric/Desktop/000_MY_WORKDIRECTORY/python/capstone yelp/3starafinn.csv',sep=',',index= False)
#bustip3star.groupby(['cat'])['sentiment3_category_adjusted'].describe()[]
#bustip3star['cat'].value_counts()


# In[466]:


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


# In[467]:


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


# In[393]:


#[i for i in bustip2star['sentiment2_score'] if i > 90]
max(bustip3star['sentiment3_scores_adjusted'])
#bustip3star[bustip3star['business_id']=='yDKmcWQ_Zycr4ekLV4CqFg']


# In[430]:


plt.figure(figsize=(20,10))
bustip3star['sentiment3_scores_adjusted'] = bustip3star['sentiment3_scores_adjusted'].astype(float)
bp = sns.boxplot(x='cat', y="sentiment3_scores_adjusted", hue='cat', data=bustip3star)


# In[ ]:


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


# In[68]:


build_text(yelprev_df1)


# In[166]:


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


# In[167]:


[swn_polarity(i) for i in yelprev_df1['text']]


# In[168]:


sentiment_polarity = [swn_polarity(i) for i in yelprev_df1['text']]
yelprev12 = pd.DataFrame([list(yelprev_df1['business_id']),list(yelprev_df1['cat']),sentiment_polarity]).T
yelprev12.columns = ['business_id','cat','sentiment_polarity']
yelprev12.groupby(['business_id','cat']).describe()


# In[471]:


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


# In[472]:


df_vaderized = vaderize(bustipreview3star, 'text')

