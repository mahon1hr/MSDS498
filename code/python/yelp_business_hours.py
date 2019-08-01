
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



def basic_details(df):
    print('Row:{}, columns:{}'.format(df.shape[0],df.shape[1]))
    k = pd.DataFrame()
    k['number of Unique value'] = df.nunique()
    k['Number of missing value'] = df.isnull().sum()
    k['Data type'] = df.dtypes
    return k


# # Import Data
business=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business.csv')
business_attributes=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_attributes.csv')
business_hours=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_business_hours.csv')
reviews=pd.read_csv('/Users/hillarymahoney/Downloads/yelp_review.csv')

demographics=pd.read_csv('/Users/hillarymahoney/Downloads/Census.csv')


# #### Create list of city values
phx_list=['Phoenix','Pheonix AZ','Phoeniix','Phoenix AZ','Phoenix metro area','Phoenix Valley','Phoenix,','Phoenix, AZ','Phoneix','Phoniex','Phx']




phx_demographics=demographics[demographics.primary_City.isin(phx_list)]
phx_business = business[business.city.isin(phx_list)]
#phx_business=business.loc[business['city']=='Phoenix']


# # Explore Yelp Data
# #### Explore Business hour data

#Replace all 'None' in business hours
cols_v = list(business_hours.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    business_hours[cols_v[i]].replace('None', np.nan, inplace=True)

basic_details(business_hours)


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

## Prepare start/finish/delta features for every weekday
bh_colnames = business_hours.columns
for c in bh_colnames[1:]:
    business_hours['{0}_s'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[0])
    business_hours['{0}_f'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[1])
    business_hours['{0}_d'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[2])
# business_hours = business_hours.drop(bh_colnames[1:], axis=1)
business_hours


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

business_hours['monday']=np.where(business_hours.monday.notnull(),1,0)
business_hours['tuesday']=np.where(business_hours.tuesday.notnull(),1,0)
business_hours['wednesday']=np.where(business_hours.wednesday.notnull(),1,0)
business_hours['thursday']=np.where(business_hours.thursday.notnull(),1,0)
business_hours['friday']=np.where(business_hours.friday.notnull(),1,0)
business_hours['saturday']=np.where(business_hours.saturday.notnull(),1,0)
business_hours['sunday']=np.where(business_hours.sunday.notnull(),1,0)



business_hours['open_weekdays'] = np.where((business_hours['monday']==1) & 
                                           (business_hours['tuesday']==1) & 
                                           (business_hours['wednesday']==1) & 
                                           (business_hours['thursday']==1),1,0)
business_hours['open_fridays']=np.where(business_hours['friday']==1,1,0)
business_hours['open_weekends']=np.where((business_hours['saturday']==1)|
                                         (business_hours['sunday']==1),1,0)

business_hours

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



#filter for restaurants
#list of key words: Restuarants, Food, Restaurant
phx_businesses=phx_business[phx_business['categories'].str.contains('Restaurants') | 
                            phx_business['categories'].str.contains('Restaurant')]
phx_businesses.shape


#drop irrelevant columns
bus_atts=business_attributes.drop(['AcceptsInsurance','HairSpecializesIn_coloring','HairSpecializesIn_africanamerican','HairSpecializesIn_curly','HairSpecializesIn_perms','HairSpecializesIn_kids','HairSpecializesIn_extensions','HairSpecializesIn_asian','HairSpecializesIn_straightperms'],axis=1)


dfs = [phx_businesses, bus_atts, business_hours, reviews] # list of dataframes

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
                                            how='left'), dfs)
df_merged=df_merged.drop(['neighborhood'],axis=1)
print(df_merged.shape)
df_merged.head(50)
# if you want to fill the values that don't exist in the lines of merged dataframe simply fill with required strings as

#df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
#                                            how='left'), data_frames).fillna('void')


