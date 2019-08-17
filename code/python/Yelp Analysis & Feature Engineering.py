
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


def basic_details(df):
    print('Row:{}, columns:{}'.format(df.shape[0],df.shape[1]))
    k = pd.DataFrame()
    k['number of Unique value'] = df.nunique()
    k['Number of missing value'] = df.isnull().sum()
    k['Data type'] = df.dtypes
    return k


# # Import Data

# In[30]:


# business=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business.csv')
business=pd.read_csv('/Users/hillarymahoney/Documents/Predict 498/Final Project/business_final.csv')
business_attributes=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_attributes.csv')
business_hours=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_hours.csv')
reviews=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_review.csv')


# In[31]:


demographics=pd.read_csv('/Users/hillarymahoney/Downloads/Census.csv')


# #### Create list of city values

# In[32]:


phx_list=['Phoenix','Pheonix AZ','Phoeniix','Phoenix AZ','Phoenix metro area','Phoenix Valley','Phoenix,','Phoenix, AZ','Phoneix','Phoniex','Phx']


# In[33]:


phx_demographics=demographics[demographics.primary_City.isin(phx_list)]
print(phx_demographics.shape)

print(phx_demographics['zcta'].nunique())


# In[34]:


phx_business = business[business.city.isin(phx_list)]
#phx_business=business.loc[business['city']=='Phoenix']
phx_business.shape


# # Explore Demographic Data

# In[35]:


phx_demographics.head()


# In[36]:


for col in phx_demographics.columns:
    print(col)


# join on zip code??

# In[37]:


phx_demographics.describe(include='all').T


# # Explore Yelp Data

# ### Explore Business hour data

# In[38]:


business_hours.head(50)


# In[39]:


#Replace all 'None' in business hours df
cols_v = list(business_hours.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    business_hours[cols_v[i]].replace('None', np.nan, inplace=True)


# In[40]:


basic_details(business_hours)


# In[41]:


## function for get time_range from string
def get_time_range(s):
    if isinstance(s, str):
        t1, t2 = s.split('-')
        h1, m1 = map(int, t1.split(':'))
        h2, m2 = map(int, t2.split(':'))
        m1, m2 = m1/60, m2/60
        t1, t2 = h1+m1, h2+m2
        if t2 < t1:
            d = t2+24-t1
        else:
            d = t2-t1
        return t1, t2, d
    else:
        return None, None, None


# In[42]:


## Prepare start/finish/delta features for every weekday
bh_colnames = business_hours.columns
for c in bh_colnames[1:]:
    business_hours['{0}_s'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[0])
    business_hours['{0}_f'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[1])
    business_hours['{0}_d'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[2])
# business_hours = business_hours.drop(bh_colnames[1:], axis=1)
business_hours


# In[43]:


#subset data for plots
f, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
sns.distplot(business_hours['mo_s'].dropna(), color='slateblue', ax=ax1, label='monday_start_times')
sns.distplot(business_hours['mo_f'].dropna(), color='slateblue', ax=ax2, label='monday_start_times')
sns.distplot(business_hours['mo_d'].dropna(), color='slateblue', ax=ax3, label='monday_start_times')

plt.show()


# In[44]:


#subset data for plots
f, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
sns.distplot(business_hours['sa_s'].dropna(), color='slateblue',ax=ax1, label='sunday_start_times')
sns.distplot(business_hours['sa_f'].dropna(), color='slateblue',ax=ax2, label='sunday_start_times')
sns.distplot(business_hours['sa_d'].dropna(), color='slateblue',ax=ax3, label='sunday_start_times')

plt.show()


# In[45]:


# wkday=business_hours.iloc[:,1:20]

# fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=[15, 9])
# for wd in [c for c in wkday.columns if '_s' in c]:  
#     sns.distplot(wkday[wd].dropna(), ax=ax1, label=wd)
# for wd in [c for c in wkday.columns if '_f' in c]:  
#     sns.distplot(wkday[wd].dropna(), ax=ax2, label=wd)
# for wd in [c for c in wkday.columns if '_d' in c]:  
#     sns.distplot(wkday[wd].dropna(), ax=ax3, label=wd)
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax1.set_title('Start hours distribution')
# ax2.set_title('Finish hours distribution')
# ax3.set_title('Duration distribution')
# plt.show()


# In[46]:


# # define weekday vs weekend categories
# weekdays = ['mo', 'tu', 'we', 'th']
# fridays = ['fr']
# weekends = ['sa', 'su']

# ## define new_cols
# bh_newcols = ['business_id']
# for wg_name, wg in zip(['weekdays', 'fridays', 'weekends'], [weekdays, fridays, weekends]):
#     for f in ['s', 'f', 'd']:
#         cols = list(map(lambda d: '{0}_{1}'.format(d,f), wg))
#         bh_newcols.append('{0}_{1}'.format(wg_name, f))
#         b_hours['{0}_{1}'.format(wg_name, f)] = b_hours.loc[:, cols].median(axis=1)

# b_hours.loc[:, bh_newcols].head(30)


# In[47]:


# weekdays = ['monday', 'tuesday', 'wednesday', 'thursday']
# fridays = ['friday']
# weekends = ['saturday', 'sunday']

business_hours['monday']=np.where(business_hours.monday.notnull(),1,0)
business_hours['tuesday']=np.where(business_hours.tuesday.notnull(),1,0)
business_hours['wednesday']=np.where(business_hours.wednesday.notnull(),1,0)
business_hours['thursday']=np.where(business_hours.thursday.notnull(),1,0)
business_hours['friday']=np.where(business_hours.friday.notnull(),1,0)
business_hours['saturday']=np.where(business_hours.saturday.notnull(),1,0)
business_hours['sunday']=np.where(business_hours.sunday.notnull(),1,0)


# In[48]:


business_hours[['monday','tuesday','wednesday','thursday','friday','saturday','sunday']].sum().plot.bar()
plt.show()


# In[49]:


business_hours['open_weekdays'] = np.where((business_hours['monday']==1) & 
                                           (business_hours['tuesday']==1) & 
                                           (business_hours['wednesday']==1) & 
                                           (business_hours['thursday']==1),1,0)
business_hours['open_fridays']=np.where(business_hours['friday']==1,1,0)
business_hours['open_weekends']=np.where((business_hours['saturday']==1)|
                                         (business_hours['sunday']==1),1,0)

business_hours


# In[50]:


#look at businesses where Fri,Sat, and Sun are all = 0 compared to Friday=1
# wknd=b_hours[['business_id','friday','saturday','sunday']]
# fri=wknd[(wknd.friday==1)&(wknd.saturday==0)&(wknd.sunday==0)]
# fri2=wknd[(wknd.friday==0)&(wknd.saturday==0)&(wknd.sunday==0)]
# print(len(wknd),len(fri),len(fri2),(100*len(fri)/len(fri2)))


# In[51]:


#break categories down further
#breakfast: 5 a.m. to 10:30 a.m.
#lunch: 11 a.m. to 3:30 p.m.
#dinner: 4 p.m. to 9:30 p.m.
#late_night: 10 p.m. to 4:30 a.m.

business_hours['weekday_breakfast']=np.where(
    (business_hours['mo_s']> 4.5)&(business_hours['mo_s']<10.5)|
    (business_hours['tu_s']> 4.5)&(business_hours['tu_s']<10.5)|
    (business_hours['we_s']> 4.5)&(business_hours['we_s']<10.5)|
    (business_hours['th_s']> 4.5)&(business_hours['th_s']<10.5)|
    (business_hours['fr_s']> 4.5)&(business_hours['fr_s']<10.5),1,0)

business_hours['weekend_breakfast']=np.where(
    (business_hours['sa_s']> 4.5)&(business_hours['sa_s']<10.5)|
    (business_hours['su_s']> 4.5)&(business_hours['su_s']<10.5),1,0)

business_hours['weekday_lunch']=np.where(
    (business_hours['mo_s']< 10.5)&(business_hours['mo_f']>15.5)|
    (business_hours['tu_s']< 10.5)&(business_hours['tu_f']>15.5)|
    (business_hours['we_s']< 10.5)&(business_hours['we_f']>15.5)|
    (business_hours['th_s']< 10.5)&(business_hours['th_f']>15.5)|
    (business_hours['fr_s']< 10.5)&(business_hours['fr_f']>15.5),1,0)

business_hours['weekend_lunch']=np.where(
    (business_hours['sa_s']< 11)&(business_hours['sa_f']>15.5)|
    (business_hours['su_s']< 11)&(business_hours['su_f']>15.5),1,0)

business_hours['weekday_dinner']=np.where(
    (business_hours['mo_s']< 15.5)&(business_hours['mo_f']>20)|
    (business_hours['tu_s']< 15.5)&(business_hours['tu_f']>20)|
    (business_hours['we_s']< 15.5)&(business_hours['we_f']>20)|
    (business_hours['th_s']< 15.5)&(business_hours['th_f']>20)|
    (business_hours['fr_s']< 15.5)&(business_hours['fr_f']>20),1,0)

business_hours['weekend_dinner']=np.where(
    (business_hours['sa_s']< 15.5)&(business_hours['sa_f']>20)|
    (business_hours['su_s']< 15.5)&(business_hours['su_f']>20),1,0)

business_hours['weekday_late_night']=np.where(
    (business_hours['mo_s']< 23.5)&(business_hours['mo_f']<4)|
    (business_hours['tu_s']< 23.5)&(business_hours['tu_f']<4)|
    (business_hours['we_s']< 23.5)&(business_hours['we_f']<4)|
    (business_hours['th_s']< 23.5)&(business_hours['th_f']<4)|
    (business_hours['fr_s']< 23.5)&(business_hours['fr_f']<4),1,0)

business_hours['weekend_late_night']=np.where(
    (business_hours['sa_s']< 23.5)&(business_hours['sa_f']<4)|
    (business_hours['su_s']< 23.5)&(business_hours['su_f']<4),1,0)

business_hours


# In[52]:


#drop cols 8:28
business_hrs=business_hours.drop(business_hours.loc[:,'mo_s':'su_d'].columns,axis=1)
business_hrs


#   

# ### Explore Business Attributes

# In[53]:


#Replace all True/False to 1/0 in Business Attributes
cols_v = list(business_attributes.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    business_attributes[cols_v[i]].replace('Na', np.nan, inplace=True)
    business_attributes[cols_v[i]].replace('True', 1, inplace=True)
    business_attributes[cols_v[i]].replace('False', 0, inplace=True)


# In[54]:


percent_missing = (business_attributes.isnull().sum() * 100 / len(business_attributes)).sort_values()
percent_missing


# In[55]:


x=percent_missing.sort_values(ascending=False)
# x=percent_missing.iloc[0:25]

#chart
plt.figure(figsize=(18,8))
ax = sns.barplot(x.index, x.values, alpha=0.8,color='slateblue')
plt.title("% of Data Missing for Yelp Attributes",fontsize=18)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.ylabel('% of missing values', fontsize=12)
plt.xlabel('Column Name', fontsize=12)

#adding the text labels
# rects = ax.patches
# for rect, label in zip(rects, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
ax.axhline(75, ls='--')
ax.text(44,75, "75%")

plt.show()


# In[56]:


max_percent_of_nas = 99.0
bus_atts = business_attributes.loc[:, (((business_attributes.isnull().sum(axis=0)*100)/len(business_attributes)) <= max_percent_of_nas)]
bus_atts=bus_atts.drop('HairSpecializesIn_coloring',axis=1)


#   

# ### Explore Business Data

# In[57]:


#How many categories do we have?
categories = pd.concat(
    [pd.Series(row['business_id'], row['categories'].split(';')) for _, row in phx_business.iterrows()]
).reset_index()
categories.columns = ['category', 'business_id']


# In[58]:


x=categories.category.value_counts()
print("There are ",len(x)," different types/categories in Phoenix")


# In[59]:


#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:25]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])
plt.title("What are the top categories?",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[60]:


#filter for restaurants
#list of key words: Restuarants, Food, Restaurant
phx_businesses=phx_business[phx_business['categories'].str.contains('Restaurants') | 
                            phx_business['categories'].str.contains('Restaurant')]
phx_businesses.shape


# In[61]:


#determine chain restaurants
phx_business['chain_restaurant'] = phx_business.duplicated(['name'])

phx_business


# In[78]:


dfs = [phx_business, bus_atts, business_hrs] # list of dataframes
# dfs = [phx_business, bus_atts, business_hrs, reviews] # list of dataframes

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
                                            how='left'), dfs)
df_merged=df_merged.drop(['neighborhood'],axis=1)
print(df_merged.shape)
df_merged.head(50)
# if you want to fill the values that don't exist in the lines of merged dataframe simply fill with required strings as

#df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
#                                            how='left'), data_frames).fillna('void')


# In[79]:


cols_v = list(df_merged.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    df_merged[cols_v[i]].replace('Na', np.nan)
    df_merged[cols_v[i]].replace('True', 1)
    df_merged[cols_v[i]].replace('False', 0)
    df_merged[cols_v[i]].replace('Chain', 1)
    df_merged[cols_v[i]].replace('Non-Chain', 0)

for col in df_merged.columns:
    print(col)


# In[80]:


df_merged


# In[81]:


final_data=df_merged.to_csv('/Users/hillarymahoney/Documents/Predict 498/Final Project/final_data.csv',index = None, header=True)


# In[64]:


# check missing data
percent_missing = (df_merged.isnull().sum() * 100 / len(df_merged)).sort_values()
# percent_missing

# isna() produces same results as isnull()
# percent_missing = (df_merged.isna().sum() * 100 / len(df_merged)).sort_values()
# drop neighborhood column
#hours have 'None' value


# In[65]:


### df_merged.describe(include='all').T


# In[66]:


## How many businesses in dataset?
print('Total restaurants:', df_merged['business_id'].nunique())


# In[67]:


np.min(df_merged['date']), np.max(df_merged['date'])


# In[68]:


# plot reviews by year
df_merged['date'] = pd.to_datetime(df_merged['date'])
df_merged['year'] = df_merged['date'].dt.year
df_merged['month'] = df_merged['date'].dt.month

f,ax = plt.subplots(1,2, figsize = (14,6))
ax1,ax2 = ax.flatten()
cnt = df_merged.groupby('year').count()['review_count'].to_frame()
sns.barplot(cnt.index, cnt['review_count'],color='slateblue', ax=ax1)

for ticks in ax1.get_xticklabels():
    ticks.set_rotation(45)

cnt = df_merged.groupby('month').count()['review_count'].to_frame()
sns.barplot(cnt.index, cnt['review_count'],color='slateblue', ax = ax2)

for ticks in ax2.get_xticklabels():
    ticks.set_rotation(45)


# In[69]:


cnt = df_merged.groupby(['month','stars_y'], as_index=False).agg({"review_count":"count"})
star_list = list(range(1,6))
cnt


# In[106]:


pandas.to_numeric([String Column])
cnt = df_merged.groupby(['month', 'stars_y'])['review_count'].count().unstack('stars_y').fillna(0)
# cnt[['abuse','nff']].plot(kind='bar', stacked=True)
cnt.columns#[1.0, 2.0, 3.0, 4.0, 5.0].plot(kind='bar', stacked=True)


# In[41]:


df_merged['review_date'] = pd.to_datetime(df_merged['date'], format='%Y%m%d')
df_merged['month_review'] = df_merged.review_date.dt.to_period('M')


# In[45]:


#Reviews per year and month

grp_date = df_merged.groupby(['review_date'])['business_id'].count()
grp_month = df_merged.groupby(['month_review'])['business_id'].count()
grp_star = df_merged.groupby(['month_review','stars_y'])['business_id'].count()

ts = pd.Series(grp_date)
ts.plot(kind='line', figsize=(20,10),title='Daily Reviews per Year', color='slateblue')
plt.title("Daily Reviews per Year",fontsize=14)
plt.y
plt.show()

ts1 = pd.Series(grp_star)
ts1.plot(kind='line', figsize=(15,8),title='Reviews per month')
plt.show()


# In[43]:


fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
df_merged.groupby(['review_date','stars_y']).count()['business_id'].unstack().plot(cmap="BuPu",ax=ax)
ax.set_xticklabels(rotation=0, fontsize=10)
ax.set_yticklabels(rotation=0, fontsize=10)


# In[47]:


#how many businesses are open?
print(df_merged['is_open'].value_counts()*100/len(df_merged))
plt.figure(figsize=(8,5))
sns.countplot(df_merged['is_open'])


# In[ ]:


grp_star


# ## Explore Zip Codes

# In[44]:


zipCode_business_counts = df_merged[['postal_code', 'business_id']].groupby(['postal_code'])['business_id'].agg('count').sort_values(ascending=False)

zipCode_business_counts.rename(columns={'business_id' : 'number_of_businesses'}, inplace=True)

zipCode_business_counts[0:25].sort_values(ascending=True).plot(kind='barh', stacked=False, figsize=[7,7], color='slateblue')
plt.title('Top 25 Zip Codes Ranked by Number of Businesses Listed',fontsize=14)


# In[45]:


# look at ratings and reviews per zip code
zipCode_business_reviews = df_merged[['postal_code', 'review_count', 'stars_x']].groupby(['postal_code']).agg({'review_count': 'sum', 'stars_x': 'mean', 'stars_x':'median'}).sort_values(by='review_count', ascending=True)


zipCode_business_reviews['review_count'][0:25].plot(kind='barh', stacked=False, figsize=[7,7],color='slateblue')
plt.title('Top 25 Zip Codes Ranked by Number of Reviews',fontsize=14)


# In[46]:


zipCode_business_reviews[zipCode_business_reviews.review_count > 5]['stars_x'][0:25].sort_values().plot(kind='barh', stacked=False, figsize=[7,7],color='slateblue')

plt.title('Top 25 Zip Codes Ranked by Average Stars',fontsize=14)


# In[50]:


zipCode_business_reviews = df_merged[['postal_code', 'review_count', 'cat']].groupby(['postal_code']).agg({'review_count': 'sum', 'cat': lambda x: x.nunique()}).sort_values(by='cat', ascending=True)


zipCode_business_reviews['cat'][0:25].plot(kind='barh', stacked=False, figsize=[7,7],color='slateblue')
plt.title('Top 25 Zip Codes Ranked by Number of Food Categories',fontsize=14)


# In[54]:


# df_merged['stars_x'].hist()
sns.countplot(x='stars',data=phx_business,color='slateblue')
plt.ylabel('frequency', fontsize=12)
plt.xlabel('Number of Stars', fontsize=12)
plt.title('Overall Star Rating per Restaurant',fontsize=14)


# In[55]:


sns.countplot(x='stars',data=reviews,color='slateblue')
plt.ylabel('frequency', fontsize=12)
plt.xlabel('Number of Stars', fontsize=12)
plt.title('Star Rating per Restaurant Review',fontsize=14)


# In[56]:


df_merged['review_count'].hist()

#try transformation like log or logit
# df_merged['review_count'].hist().set_yscale('log')


# In[ ]:


#add a column to the data set
df_merged['log_reviews']=np.log(df_merged['review_count']+0.1)


# In[ ]:


df_merged['log_reviews'].hist()


# In[51]:


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

# In[52]:


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


# In[53]:


#rearrange data to use folium for mapping
df=[]
list_stars=list(ratings['stars_x'].unique())

for star in list_stars:
    subset=ratings_phx[ratings_phx['stars_x']==star]
    df.append(subset[['latitude','longitude']].values.tolist())


# In[54]:


#use central points from scatter plot above
zoom_start=11
print("Phoenix Heat Map of Businesses")
m=folium.Map(location=[lat,lon],tiles="OpenStreetMap",zoom_start=zoom_start)

heat_map=plugins.HeatMapWithTime(df, max_opacity=0.4, auto_play=True, display_index=True, radius=7)
heat_map.add_to(m)
m


# Is the restaurant part of a chain? If the restaurant name appears more than once in the list then it is considered to be part of a chain. This includes national or local chains. Some chains that are represented by only one restaurant in the particular list did not count as a chain due to the way a chain is defined.
# 
# What is the local restaurant density? Based on the restaurant coordinates, I created a list of restaurants within 1 mile radius for each of the restaurants in the list.
# 
# What is the review count, star rating and price (i.e. general dining cost) relative to surrounding restaurants? The surrounding restaurants within 1 mile radius of each restaurant were identified (similar to the restaurant density calculation) and the relative values for the review count, star rating and price of each restaurant were calculated by subtracting the mean of this group of restaurants from each individual restaurant and dividing with the standard deviation of the value for this group of restaurants.
# 
# What is each restaurant’s age? This value is approximated by the date of the first yelp review. This means that restaurants that joined yelp late or do not receive frequent comments would appear to have a relatively younger age than their real value. Also, the restaurant age is limited by the date Yelp was founded (i.e. 2004).
# 

# In[ ]:


ambiencelist = df_merged.filter(like='Ambience').columns.tolist()
y = int(len(ambiencelist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(df_merged[ambiencelist[0]], ax=ax[i,j], palette="Set1")
        del ambiencelist[0]
fig.tight_layout()        


# In[ ]:


bplist = yelp_attr.filter(like='BusinessParking').columns.tolist()
y = int(len(bplist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(yelp_attr[bplist[0]], ax=ax[i,j], palette="Set1")
        del bplist[0]
fig.tight_layout()        


# In[ ]:


bnlist = yelp_attr.filter(like='BestNights').columns.tolist()
y = int(len(bnlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(yelp_attr[bnlist[0]], ax=ax[i,j], palette="Set1")
        del bnlist[0]
fig.tight_layout()        


# In[ ]:


meallist = yelp_attr.filter(like='GoodForMeal').columns.tolist()
y = int(len(meallist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[meallist[0]], ax=ax[i,j], palette="Set1")
        del meallist[0]
fig.tight_layout()        


# In[ ]:


dtlist = yelp_attr.filter(like='DietaryRestrictions').columns.tolist()
del dtlist[0]
y = int(len(dtlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[dtlist[0]], ax=ax[i,j], palette="Set1")
        del dtlist[0]
fig.tight_layout()        


# ### look for variables that affect or are related to stars and positive reviews..
# what makes a positive review...

# In[ ]:


#!pip install squarify
import squarify
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re 
# !pip install gensim
import gensim 
from gensim import corpora


# In[ ]:


# Meta feature of text
tips['num_words'] = tips['text'].str.len()
tips['num_uniq_words'] = tips['text'].apply(lambda x: len(set(str(x).split())))
tips['num_chars'] = tips['text'].apply(lambda x: len(str(x)))
tips['num_stopwords'] = tips['text'].apply(lambda x: len([w for w in str(x).lower().split() 
                                                      if w in set(stopwords.words('english'))]))


