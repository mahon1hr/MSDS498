library(lattice) # required for the xyplot() function
library(stargazer) #use this package to create nice table outputs
library(rpart) 
library(rpart.plot)
library(tree)
library(rattle)
library(caTools)
library(ggplot2)
library(dplyr)
library(leaps)
library(MASS)
library(tidyverse)
library(caret)
library(randomForest)
library(dplyr)

#Set working directory
setwd("C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets")


############################ Modeling by Business ID ##############################

# Read in dataset for Regression
yelp <- read.csv("business_Linear_AllDummyNoCensus.csv",na.strings="NULL")
#dim(yelp)

# Define output path for html files;
out.path <- 'C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/';

file.name <- 'SummaryStats.html';
stargazer(yelp, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 1.2: Summary Statistics for Yelp Dataset'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)

#See if any missing values exist
sum(is.na(yelp))

#Model Tree
tree.yelp=tree(stars~.,data=yelp[,-c(1)])

#Sumarry to get variables used and SSE
summary(tree.yelp)
# Variables actually used in tree construction:
#   [1] "chain"              "friday"             "Fast_Food"          "thursday"          
# [5] "weekday_late_night" "saturday"          
# Number of terminal nodes:  7 
# Residual mean deviance:  0.495 = 1604 / 3240 

#Display tree structure
plot(tree.yelp)
text(tree.yelp,pretty=0) #includes category names

#####################
# Train / Test Split
#####################

yelpsample = sample.split(yelp,SplitRatio = 0.75) 
yelp_train =subset(yelp,yelpsample ==TRUE) 
yelp_test=subset(yelp, yelpsample==FALSE) 


##################### (remove cluster)
# Just on zip code binary, category binary, hours binary
# could reduce hours

#Regular linear regression on training set
#Removed census variables because they are correlated
#Removed target leak variables: review counts
regmodel=lm(stars~.,data=yelp_train[,-c(1)])
summary(regmodel)

#RMSE
sqrt(mean(regmodel$residuals^2))

# Run model on test data
testmodel = predict(regmodel,newdata=yelp_test[,-c(1)])
output <- cbind(yelp_test, floor(testmodel*2) / 2)

# Create Output File
write.csv(output, file = "LinearResults.csv")


############################ Modeling by Category ##############################
# Average all continuous variables by Category
# Use Census data instead of Zip code here
category <- read.csv("business_byCategory.csv",na.strings="NULL")
#dim(category)


#############################################
#Handle Cardinality issues with Cat field
#############################################

# Create Collapsed Category Function: Assigns as "Other"
collapsecategory <- function(x, p) {
  levels_len = length(levels(x))
  levels(x)[levels_len+1] = 'Other'
  y = table(x)/length(x)
  y1 = as.vector(y)
  y2 = names(y)
  y2_len = length(y2)
  for (i in 1:y2_len) {
    if (y1[i]<=p){
      x[x==y2[i]] = 'Other'
    }
  }
  x <- droplevels(x)
  x
}

# Create list of variables with cardinality issues
cardinal <- list('cat')

# Create function that loops through list
for(i in cardinal){
  
  # Get Total counts demonimator for those records without missing values. Convert to integer.        
  total <- as.integer(category %>%
                        summarise(n()))
  
  # Create new dataframe with variable and convert to Local Data Frame
  local_df <- tbl_df(dplyr::select(category, i, business_id))
  
  # For each variable entry, calculate the % total and sort in descending order
  grouped_df <- local_df %>%
    group_by_at(1) %>%
    summarise(perc_tot = n()/total) %>%
    arrange(desc(perc_tot))
  
  # Rank by percent total and create new variable
  ranked_df <- grouped_df %>% mutate(rank=row_number())
  
  # Count the number of rows. 
  cat_count <- nrow(ranked_df)
  
  # If the number of categories (rows) is less than 20, move on. 
  # Otherwise, grab the percentile at rank 20 and use that as a cut off point for "Other" category
  if (cat_count > 20) {
    
    # Get percent total value at rank 20 as numeric 
    filtered <- filter(ranked_df, rank==20)
    perc20 <- as.numeric(dplyr::select(filtered,perc_tot))
    
    # Apply category function to those categories over rank 20
    # Replace original variable with collapsed variable
    category[i] <- collapsecategory(category[[i]],perc20)
    
  } else {
    
    print('looping through')
    
  }
}

# Check number of categories
str(category$cat) #19

# See top 10 categories
grouped_df


#####################
# Train / Test Split
#####################

catsample = sample.split(category,SplitRatio = 0.50) 
cat_train =subset(category,catsample ==TRUE) 
cat_test=subset(category, catsample==FALSE) 

###########################################
# Collapse by Category: for train and test
###########################################

# Convert to Local Data Frame, remove business id
local_df_train <- tbl_df(cat_train[,-c(1)])
local_df_test <- tbl_df(cat_test[,-c(1)])

# Calculate mean for all variables, group by category
group_cat_train <- local_df_train %>%
  group_by(cat) %>%
  summarise_all(mean, na.rm=TRUE)

group_cat_test <- local_df_test %>%
  group_by(cat) %>%
  summarise_all(mean, na.rm=TRUE)

cat_avgs_train <- as.data.frame(group_cat_train)
cat_avgs_test <- as.data.frame(group_cat_test)

# Export to csv for others to use
write.csv(cat_avgs_train, file = "CollapsedCat_Train.csv")
write.csv(cat_avgs_test, file = "CollapsedCat_Test.csv")

#####################
# Modeling
#####################



