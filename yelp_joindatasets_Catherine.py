
#import packages
import pandas as pd 


# Step 1: Import all files
business=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_useV3.csv')
business_attributes=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_attributes.csv')
business_hours=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_business_hours.csv')
reviews=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/yelp_review.csv')
demographics=pd.read_csv('C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/Census.csv')

# Step 2: Data Prep

######## Business file ##########
# This is the cleaned up file Richa put together that only contains businesses
# in the Phoenix zip code list and restaurants

### Inspect dataset ###
business.shape #(3568, 13)
business.count() #missing neighborhood field
business.head(3)
# Name: Remove Quotes before/after
# Address: Remove Quotes before/ after, 42 NA's consisting of ""
# Neighborhood field is all null, needs to be removed

### Clean up ###

# Remove Quotes before/after in Name and Address field
business.name = business.name.str.replace('"', '')
business.address = business.address.str.replace('"', '')

# Remove neighborhood field
# Also removed address, city, or state fields from analysis
business=business.drop(['neighborhood'],axis=1)
business_df = business.drop(['address','city','state'],axis=1)
business_df.shape #(3568, 9)

# Remove those entries without business_ids
#keep = business_df[business_df['business_id']!='#NAME?']
#business_df = keep[keep['name']!='Green Tea Chinese Restaurant'] #business_id = #VALUE!
#business_df.shape #(3584, 9)

# Create list of Business Id's
business_ids = business_df.business_id.unique().tolist()
len(business_ids) #3568

# Create list of Unique Zip Codes
zipcodes = business_df.postal_code.unique().tolist()
len(zipcodes) # 41

######## Business attribute file ##########

### Initial Cleanup ###

# First, remove columns Hilary already determined weren't of use
attributes_df=business_attributes.drop(['AcceptsInsurance','HairSpecializesIn_coloring','HairSpecializesIn_africanamerican','HairSpecializesIn_curly','HairSpecializesIn_perms','HairSpecializesIn_kids','HairSpecializesIn_extensions','HairSpecializesIn_asian','HairSpecializesIn_straightperms'],axis=1)

# Next only grab those business ids that are in business_ids list
attributes_df=attributes_df[attributes_df.business_id.isin(business_ids)]

### Inspect dataset ###
attributes_df.count() #3538 #30 business id's missing
attributes_df.head(10) #Nulls are listed as Na

# Replace values with "Na" with blanks
attributes_df = attributes_df.replace(to_replace = 'Na', value = "") 
attributes_df.head(10)

# Replace TRUE wtih 1 and FALSE with 0
#attributes_df = attributes_df.replace(to_replace = 'TRUE', value = 1)
#attributes_df = attributes_df.replace(to_replace = 'FALSE', value = 0)
#attributes_df.isnull()


######## Business attribute file ##########

### Inspect dataset ###
business_hours.shape #(174567, 8)
business_hours.count()
business_hours.head() #nulls are listed as None

# Only grab those business ids that are in business_ids list
hours_df=business_hours[business_hours.business_id.isin(business_ids)]
hours_df.shape #(3568, 8) #appear to have business hours for all business ids

# make sure those are unique business id's
len(hours_df.business_id.unique().tolist()) #yes, all unique

# Still need to handle NULLS


######## reviews file ##########

### Inspect dataset ###
reviews.count() #Unclear on how NA's present themselves
#review_id      5261668
#user_id        5261668
#business_id    5261668
#stars          5261668
#date           5261668
#text           5261668
#useful         5261668
#funny          5261668
#cool           5261668
reviews.shape #(5261668, 9)
reviews.head() 

# Only grab those business ids that are in business_ids list
reviews_df=reviews[reviews.business_id.isin(business_ids)]
reviews_df.shape #(323750, 9)

#Count distinct business ids
len(reviews_df.business_id.unique().tolist()) #3568 (all business ids appear to have reviews)

#Analyze using text analytics before joining to main dataset

######## demographics file ##########

### Inspect dataset ###
demographics.count()
demographics.shape #(32977, 122)
demographics.head() #nulls might be non existant or they might be indicated with 0

# Create dataframe of demographics only with zip codes from business file
demographics_df=demographics[demographics.zcta.isin(zipcodes)]
demographics_df.shape #(41, 122) - matched all 41 zip codes

# Rename zip code column
demographics_df = demographics_df.rename(columns={'zcta': 'postal_code'})
demographics_df.count()


# Step 3: Combine dataframes

# Review shapes
business_df.shape #(3568, 9)
attributes_df.shape #(3538, 73)
hours_df.shape #(3568, 8)
demographics_df.shape #(41, 122)

# Merge with Left Join
combined = pd.merge(business_df,demographics_df,how='left',on='postal_code')
combined.shape #(3568, 130) looks good

combined2 = pd.merge(combined,hours_df,how='left',on='business_id')
combined2.shape #(3568, 137) looks good

# Keep attributes on the end because they likely won't make the modeling step
# given the number of missing values
final = pd.merge(combined2,attributes_df,how='left',on='business_id')
final.shape #(3568, 209) looks good


# Step 4: Export to csv
final.to_csv(r'C:\Users\cscanlon\Desktop\SQL\NW\12-CAPSTONE\Project\Datasets\final.csv')

