#!/usr/bin/env python
# coding: utf-8

# #### Import packages and data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import time

from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats 


# In[2]:


business=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_useV3.csv')
demographics=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/Census.csv')


# In[3]:


business.head(2)


# In[ ]:


#### Generate new column for single category


# In[4]:


business['cat'] = "NA"


# #### Generate Dummy Variable Columns

# In[5]:


list_dv = ['Fast Food','Bars','Gluten-Free','Vegetarian','Vegan','Pizza']


# In[6]:


for i in list_dv: 
    business[i] = business['categories'].str.contains(i, regex=False)


# In[7]:


business.head(2)


# #### Filter by restaurants

# In[8]:


business['temp'] = "0"
business['temp'] = business['categories'].str.contains('Restaurant', regex=False)
business = business[business.temp == True]
business['temp'].value_counts()


# #### Categorize a restaurant based on categories

# In[9]:


list_cat = ['Nightlife','Music Venues','Venues & Event Spaces','Arts & Entertainment','Grocery',
            
            #######
            
            #generic categories go first
            'Comfort Food',
            'Bars',
            'Fast Food',
            'Breakfast & Brunch',
            'Desserts',           
            'Cafes','Bakeries','Delis','Cafeteria',
            'American (Traditional)',
            'American (New)',
            'Seafood',
            'Buffets',
            'Gluten-Free','Vegetarian', 'Vegan',
            'Cheesesteaks','Chicken',
            'Hot Dogs',
            
            #Filter through American food
            'Sandwiches','Burgers','Sports Bars','Barbeque','Steakhouses',
            #Filter through central American food
            'Latin America','Mexican','Tex-Mex','Peruvian','Salvadoran',
            #Filter through Mediterranean food
            'Mediterranean','Greek',
            #Filter through Asian Food
            'Chinese','Asian Fusion','Thai','Vietnamese','Japanese','Sushi',
            'Korean',
            #Filter through Middle East
            'Middle Eastern',
            
            #Filter through pizza and Italian
            'Pizza','Italian',
            #Filter through Carribean
            'Carribean','Caribbean','Cajun','Cuban',
            #Filter through European 
            'Modern European','French',
            'Russian','Ukranian','Polish','Uzbek','Afghan',
            'German','Bavarian',
            'Spanish',
            #Filter through African
            'African','Ethiopian',
            #Filter throuh Indian
            'Indian',#'Pakistani',
                     
            #Filter through Southern
            'Southern','Diners','Soul Food',
            #Filter through bar restaurants
            'Sports Bar','Wine Bars','Gastropub','Cocktail Bars','Dive Bar',
            #Filter through misc.
            'Coffee & Tea','Juice Bars & Smoothies','Ice Cream & Frozen Yogurt',
            'Bagels', 'Donuts',
            'Fish & Chips','Kosher','Pretzels',
            'Hawaiian', 
            
            ########
            
            #Label things that are not restaurants
            
            'Festival','Food Trucks','Event Planning & Services','Grocery', 
            'Food Delivery Services','Shopping','Active Life', 'Party & Event Planning',
            'Health Markets','Convenience Stores', 'Arcades','Professional Services',
            'Home & Garden','Dance Clubs', 'Wholesalers','Restaurant Supplies',
            'Automotive','Furniture Stores', 'Health & Medical','Gas Stations',
            'Kitchen & Bath','Beauty & Spas', 'Home Services','Appliances','Day Spas',
            'Personal Chefs','Caterers', 'Fashion','Food Court','Veterinarians'
            
             ]


# In[10]:


for i in list_cat: 
    business['temp'] = "0"
    business['temp'] = business['categories'].str.contains(i, regex=False)
    business['cat'] = np.where(business['temp'] == True, i, business['cat'])


# In[11]:


business['cat'].value_counts()


# #### Remove non-restaurant categories that still remain

# In[12]:


list_drop = ['Appliances','Home Services','Festival','Health & Medical',
             'Gas Stations','Day Spas','Restaurant Supplies',
             'Personal Chefs','Veterinarians','Arts & Entertainment',
             'Kitchen & Bath','Nightlife','Furniture Stores','Beauty & Spas',
             'Arcades','Dance Clubs','Party & Event Planning',
             'Convenience Stores','Caterers','Food Trucks','Active Life',
             'Food Delivery Services','Health Markets','Shopping','Grocery',
             'Event Planning & Services','Fashion'
            ]


# In[13]:


for i in list_drop:
    business = business.loc[business['cat'] != i]


# In[14]:


business.shape[0]


# In[15]:


pd.set_option('display.max_row', 100)
business['cat'].value_counts()


# #### Categorize a restaurant by its name

# In[16]:


business['temp'] = business['name'].str.contains('Burger King', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Burger', business['cat'])


# In[17]:


business['temp'] = business['name'].str.contains('McDonalds', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Burger', business['cat'])


# In[18]:


business['temp'] = business['name'].str.contains('Dairy Queen', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Ice Cream & Frozen Yogurt', business['cat'])


# In[19]:


business['temp'] = business['name'].str.contains('Noodles & Company', regex=False)
business['cat'] = np.where(business['temp'] == True, 'American (New)', business['cat'])


# In[20]:


business['temp'] = business['name'].str.contains('The Loaded Potato', regex=False)
business['cat'] = np.where(business['temp'] == True, 'American (New)', business['cat'])


# In[21]:


business['temp'] = business['name'].str.contains('Food Court', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Food Court', business['cat'])


# In[22]:


business['temp'] = business['name'].str.contains('Sonic', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Ice Cream & Frozen Yogurt', business['cat'])


# In[23]:


business['temp'] = business['name'].str.contains('Mighty Mikes', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Burger', business['cat'])


# In[24]:


business['temp'] = business['name'].str.contains('Salad', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Salad', business['cat'])


# In[25]:


business['temp'] = business['name'].str.contains('Pizza', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Pizza', business['cat'])


# In[26]:


business['temp'] = business['name'].str.contains('pizza', regex=False)
business['cat'] = np.where(business['temp'] == True, 'pizza', business['cat'])


# In[27]:


business['temp'] = business['name'].str.contains('Taco', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Mexican', business['cat'])


# In[28]:


business['temp'] = business['name'].str.contains('taco', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Mexican', business['cat'])


# In[29]:


business['temp'] = business['name'].str.contains('Carl', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Burgers', business['cat'])


# In[30]:


business['temp'] = business['name'].str.contains('Red Robin', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Burgers', business['cat'])


# In[31]:


business['temp'] = business['name'].str.contains(r'(?=.*Jack)(?=.*Box)',regex=True)
business['cat'] = np.where(business['temp'] == True, 'Burgers', business['cat'])


# In[32]:


business['temp'] = business['name'].str.contains(r'(?=.*Jack)(?=.*box)',regex=True)
business['cat'] = np.where(business['temp'] == True, 'Burgers', business['cat'])


# In[33]:


business['temp'] = business['name'].str.contains('Chicken', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Chicken', business['cat'])


# In[34]:


business['temp'] = business['name'].str.contains('Fish', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Seafood', business['cat'])


# In[35]:


business['temp'] = business['name'].str.contains('Fish & Chips', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Fish & Chips', business['cat'])


# In[36]:


business['temp'] = business['name'].str.contains('Wong', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Chinese', business['cat'])


# In[37]:


business['temp'] = business['name'].str.contains('Grill', regex=False)
business['cat'] = np.where(business['temp'] == True, 'American (New)', business['cat'])


# In[38]:


business['temp'] = business['name'].str.contains('Cafe', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Cafe', business['cat'])


# In[39]:


business['temp'] = business['name'].str.contains('Deli', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Deli', business['cat'])


# In[40]:


business['temp'] = business['name'].str.contains('Pretzels', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Pretzels', business['cat'])


# In[41]:


business['temp'] = business['name'].str.contains('Nueva', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Mexican', business['cat'])


# In[42]:


business['temp'] = business['name'].str.contains('Pueblo', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Mexican', business['cat'])


# In[43]:


business['temp'] = business['name'].str.contains('Salsita', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Mexican', business['cat'])


# In[44]:


pd.set_option('display.max_row', 100)
business['cat'].value_counts()


# #### Consolidate categories

# In[45]:


business['temp'] = business['cat'].str.contains('Bavarian', regex=False)
business['cat'] = np.where(business['temp'] == True, 'German', business['cat'])


# In[46]:


business['temp'] = business['cat'].str.contains('Salvadoran', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Latin America', business['cat'])


# In[47]:


business['temp'] = business['cat'].str.contains('Peruvian', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Latin America', business['cat'])


# In[48]:


business['temp'] = business['cat'].str.contains('Ethiopian', regex=False)
business['cat'] = np.where(business['temp'] == True, 'African', business['cat'])


# In[49]:


business['temp'] = business['cat'].str.contains('Cheesesteaks', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Sandwiches', business['cat'])


# In[50]:


business['temp'] = business['cat'].str.contains('pizza', regex=False)
business['cat'] = np.where(business['temp'] == True, 'Pizza', business['cat'])


# In[51]:


business['cat'].value_counts()


# In[ ]:


#### National chains (https://www.restaurantbusinessonline.com/top-500-chains)


# In[214]:


business['chain'] = False
business['chain'] = business.groupby(['name']).transform('count')


# In[215]:


business['chain'] = np.where(business['chain'] > 1, True, False)


# In[216]:


business['Fast Food'].value_counts()


# In[217]:


business['Bars'].value_counts()


# In[218]:


business['Gluten-Free'].value_counts()


# In[219]:


business['Vegetarian'].value_counts()


# In[220]:


business['Vegan'].value_counts()


# In[221]:


business['Pizza'].value_counts()


# In[222]:


business['chain'].value_counts()


# In[223]:

## FOLLOWING CODE ADDED BETWEEN V4 AND V5 BY T. MARASS
# Code to create dummy variable columns for each of the restaurant categories

list_cat = business['cat'].unique()
for i in list_cat:
    j = i + '_dum'
    business[j] = np.where(business['cat'] == i, True, False)

## PRECEEDING CODE ADDED BETWEEN V4 AND V5 BY T. MARASS

business.head(20)
business.to_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/business.csv',sep=',',index= False)


# #### Use the code below to interrogate categories

# In[224]:


i = False #write in what category you are interested in


# In[225]:


bizzy = business.loc[business['chain']==i]
pd.set_option('display.max_row', 100)
bizzy['name'].value_counts() ==1


# In[226]:


bizzy.head(20)


# In[227]:


business_attributes=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_attributes.csv')
business_hours=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_hours.csv')


# In[228]:


def basic_details(df):
    print('Row:{}, columns:{}'.format(df.shape[0],df.shape[1]))
    k = pd.DataFrame()
    k['number of Unique value'] = df.nunique()
    k['Number of missing value'] = df.isnull().sum()
    k['Data type'] = df.dtypes
    return k


# In[229]:


cols_v = list(business_hours.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    business_hours[cols_v[i]].replace('None', np.nan, inplace=True)

#business_hours.nunique()


# In[230]:


#basic_details(business_hours)


# In[231]:






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


# In[232]:


bus_atts=business_attributes.drop(['AcceptsInsurance','HairSpecializesIn_coloring','HairSpecializesIn_africanamerican',
                                   'HairSpecializesIn_curly','HairSpecializesIn_perms','HairSpecializesIn_kids',
                                   'HairSpecializesIn_extensions','HairSpecializesIn_asian',
                                   'HairSpecializesIn_straightperms'],axis=1)


# In[233]:


dfs = [business, bus_atts, business_hours] # list of dataframes

df_merged = pd.merge(business,bus_atts,how='left',on='business_id')
df_merged = pd.merge(df_merged,business_hours,how='left',on='business_id')

#df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['business_id'],
 #                                           how='left'), dfs)
df_merged=df_merged.drop(['neighborhood'],axis=1)
print(df_merged.shape)
df_merged.head(50)


# In[234]:


df_merged.to_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/business.csv',sep=',',index= False)


# In[235]:


zipcodes = business.postal_code.unique().tolist()
len(zipcodes) # 41


# In[236]:


demographics.count()
demographics.shape #(32977, 122)
demographics.head() #nulls might be non existant or they might be indicated with 0

# Create dataframe of demographics only with zip codes from business file
demographics_df=demographics[demographics.zcta.isin(zipcodes)]
demographics_df.shape #(41, 122) - matched all 41 zip codes

# Rename zip code column
demographics_df = demographics_df.rename(columns={'zcta': 'postal_code'})
demographics_df.count()


# In[237]:


business_df = pd.merge(df_merged,demographics_df,how='left',on='postal_code')


# In[238]:


business_df


# In[239]:

# Clean up data: Data type conversions and handling missing variables

# Replacing blanks/NAs with python readable NaN
cols_v = list(business_df.columns.values)[1:]
for i in range(len(cols_v)):
    #print(cols_v[i])
    business_df[cols_v[i]].replace('Na', np.nan, inplace=True)
    business_df[cols_v[i]].replace('True', 1, inplace=True)
    business_df[cols_v[i]].replace('False', 0, inplace=True)
    

# Remove attribute variables due to too many missing values, with the exception of parking
business_df.drop(business_df.iloc[:, 96:161], inplace=True, axis=1) 

business_df.drop(['ByAppointmentOnly' 
                  ,'BusinessAcceptsCreditCards'], axis=1, inplace=True) 

# Create one variable for parking based on 5 parking variables
business_df['has_parking'] = np.where((business_df['BusinessParking_garage']==1) |
           (business_df['BusinessParking_street']==1) |
           (business_df['BusinessParking_validated']==1) |
           (business_df['BusinessParking_lot']==1) |
           (business_df['BusinessParking_valet']==1),1,0)

# drop original parking variables
business_df.drop(['BusinessParking_garage'
                  ,'BusinessParking_street'
                  ,'BusinessParking_validated'
                  ,'BusinessParking_lot'
                  ,'BusinessParking_valet'], axis=1, inplace=True)
    
# Remove other unnecessary variables
business_df.drop(['categories' # Original variable not needed
                  ,'cat' # Categorical version of this variable not needed now
                  ,'pop_Est_Factor' #All one value
                  ,'pop_Est_Level' # All list "City"
                  ,'primary_City' #unnecessary census data
                  ,'state_y' #unnecessary census data
                  ,'primary_County' #unnecessary census data
                  ,'latLong'], axis=1, inplace=True) # Don't believe we need this as we already have the lat/long of the business   
    
    

business_df.to_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/business.csv',sep=',',index= False)


# In[ ]:




