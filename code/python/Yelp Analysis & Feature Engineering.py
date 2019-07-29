
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('precision', 4)

import numpy as np
import math
import itertools
import seaborn as sns
import time

from glob import glob
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from scipy import stats
import gc

import folium
import folium.plugins as plugins


# In[2]:


business=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business.csv')
business_attributes=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_attributes.csv')
business_hours=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_hours.csv')
reviews=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_review.csv')


# In[3]:


demographics=pd.read_csv('/Users/hillarymahoney/Downloads/Census.csv')


# In[4]:


phx_list=['Phoenix','Pheonix AZ','Phoeniix','Phoenix AZ','Phoenix metro area','Phoenix Valley','Phoenix,','Phoenix, AZ','Phoneix','Phoniex','Phx']


# In[5]:


phx_demographics=demographics[demographics.primary_City.isin(phx_list)]
phx_demographics.shape


# In[6]:


phx_business = business[business.city.isin(phx_list)]
#phx_business=business.loc[business['city']=='Phoenix']
phx_business.shape


# ## Explore Demographic Data

# In[23]:


phx_demographics.head()


# In[22]:


for col in phx_demographics.columns:
    print(col)


# join on zip code??

# In[24]:


phx_demographics.describe(include='all').T


# ## Explore Yelp Data

# In[7]:


#How many categories do we have?
categories = pd.concat(
    [pd.Series(row['business_id'], row['categories'].split(';')) for _, row in phx_business.iterrows()]
).reset_index()

categories.columns = ['category', 'business_id']


# In[8]:


fig, ax = plt.subplots(figsize=[5,10])
sns.countplot(data=categories[categories['category'].isin(
    categories['category'].value_counts().head(50).index)],
                              y='category', ax=ax)
plt.show()


# In[9]:


#filter for restaurants
#list of key words: Restuarants, Food, Restaurant
phx_businesses=phx_business[phx_business['categories'].str.contains('Restaurants') | 
                            phx_business['categories'].str.contains('Restaurant')]
phx_businesses.shape


# In[10]:


#drop irrelevant columns
bus_atts=business_attributes.drop(['AcceptsInsurance','HairSpecializesIn_coloring','HairSpecializesIn_africanamerican','HairSpecializesIn_curly','HairSpecializesIn_perms','HairSpecializesIn_kids','HairSpecializesIn_extensions','HairSpecializesIn_asian','HairSpecializesIn_straightperms'],axis=1)


# In[11]:


dfs = [phx_businesses, bus_atts, business_hours, reviews] # list of dataframes

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
                                            how='left'), dfs)
print(df_merged.shape)
df_merged.head()
# if you want to fill the values that don't exist in the lines of merged dataframe simply fill with required strings as

#df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
#                                            how='left'), data_frames).fillna('void')


# In[12]:


for col in df_merged.columns:
    print(col)


# In[13]:


df_merged.describe(include='all').T


# In[14]:


# check missing data
percent_missing = (df_merged.isnull().sum() * 100 / len(df_merged)).sort_values()
percent_missing
# isna() produces same results as isnull()
# percent_missing = (df_merged.isna().sum() * 100 / len(df_merged)).sort_values()
# drop neighborhood column


# In[15]:


## How many businesses in dataset?
print('Total restaurants:', df_merged['business_id'].nunique())


# In[16]:


np.min(df_merged['date']), np.max(df_merged['date'])


# In[18]:


# plot reviews by year
df_merged['date'] = pd.to_datetime(df_merged['date'])
df_merged['year'] = df_merged['date'].dt.year
df_merged['month'] = df_merged['date'].dt.month

f,ax = plt.subplots(1,2, figsize = (14,6))
ax1,ax2 = ax.flatten()
cnt = df_merged.groupby('year').count()['review_count'].to_frame()
sns.barplot(cnt.index, cnt['review_count'],palette = 'Blues', ax=ax1)

for ticks in ax1.get_xticklabels():
    ticks.set_rotation(45)

cnt = df_merged.groupby('month').count()['review_count'].to_frame()
sns.barplot(cnt.index, cnt['review_count'],palette = 'Purples', ax = ax2)



# In[19]:


#how many businesses are open?
print(df_merged['is_open'].value_counts())
plt.figure(figsize=(8,5))
sns.countplot(df_merged['is_open'])


# ## Explore Zip Codes

# In[20]:


zipCode_business_counts = df_merged[['postal_code', 'business_id']].groupby(['postal_code'])['business_id'].agg('count').sort_values(ascending=False)

zipCode_business_counts.rename(columns={'business_id' : 'number_of_businesses'}, inplace=True)

zipCode_business_counts[0:25].sort_values(ascending=True).plot(kind='barh', stacked=False, figsize=[10,10], color='green')
plt.title('Top 25 Zip Codes by businesses listed')


# In[21]:


# look at ratings and reviews per zip code
zipCode_business_reviews = df_merged[['postal_code', 'review_count', 'stars_x']].groupby(['postal_code']).agg({'review_count': 'sum', 'stars_x': 'mean', 'stars_x':'median'}).sort_values(by='review_count', ascending=True)


zipCode_business_reviews['review_count'][0:25].plot(kind='barh', stacked=False, figsize=[10,10],color='green')
plt.title('Top 25 Zip Codes by reviews')


# In[26]:


zipCode_business_reviews[zipCode_business_reviews.review_count > 5]['stars_x'][0:25].sort_values().plot(kind='barh', stacked=False, figsize=[10,10],color='green')

plt.title('Zip Codes with greater than 5 reviews ranked by average stars')


# In[27]:


df_merged['stars_x'].hist()


# In[28]:


#df_merged['review_count'].hist()

#try transformation like log or logit
df_merged['review_count'].hist().set_yscale('log')


# In[29]:


#add a column to the data set
df_merged['log_reviews']=np.log(df_merged['review_count']+0.1)


# In[30]:


df_merged['log_reviews'].hist()


# In[31]:


#geographically plot businesses and ratings
ratings=df_merged[['stars_x','review_count','latitude','longitude']]
#normalize data or create some sort of metric??
ratings['stars_per_100_reviews']=ratings['stars_x']*ratings['review_count']/100
ratings['popularity']=ratings['stars_x']*ratings['review_count']

ratings


# One of the reasons that it's easy to get confused between scaling and normalization is because the terms are sometimes used interchangeably and, to make it even more confusing, they are very similar! In both cases, you're transforming the values of numeric variables so that the transformed data points have specific helpful properties. The difference is that, in scaling, you're changing the range of your data while in normalization you're changing the shape of the distribution of your data.
# 
# This means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures of how far apart data points, like support vector machines, or SVM or k-nearest neighbors, or KNN. With these algorithms, a change of "1" in any numeric feature is given the same importance.
# 
# Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.
# 

# In[32]:


f, ax1=plt.subplots(1,1,figsize=(15,7))

#pick central point in Phoenix
lat = 33.435463
lon = -112.006989

lon_min, lon_max = lon-0.3, lon+0.5
lat_min, lat_max = lat-0.4, lat+0.5

ratings_phx=ratings[(ratings["longitude"]>lon_min) & (ratings["longitude"]<lon_max) & (ratings["latitude"]>lat_min) & (ratings["latitude"]<lat_max)]

ratings_phx.plot(kind='scatter',x='longitude',y='latitude', color='red',s=0.05, alpha=0.8, subplots=True, ax=ax1)
ax1.set_title("Phoenix")
f.show()


# In[33]:


#rearrange data to use folium for mapping
df=[]
list_stars=list(ratings['stars_x'].unique())

for star in list_stars:
    subset=ratings_phx[ratings_phx['stars_x']==star]
    df.append(subset[['latitude','longitude']].values.tolist())


# In[35]:


#use central points from scatter plot above
zoom_start=11
print("Phoenix Heat Map of Businesses")
m=folium.Map(location=[lat,lon],tiles="OpenStreetMap",zoom_start=zoom_start)

heat_map=plugins.HeatMapWithTime(df, max_opacity=0.4, auto_play=True, display_index=True, radius=7)
heat_map.add_to(m)
m


# ### look for variables that affect or are related to stars and positive reviews..
# what makes a positive review...
