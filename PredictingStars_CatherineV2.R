# Load Packages
library(stargazer)
library(dplyr)
library(epiDisplay) #tab1 function in EDA
library(imputeTS) #missing values 
library(caTools) #train test split
library(pacman)
require(ggplot2)
require(rpart)
require(rpart.plot)
require(tree)
require(rattle)
require(ROCR)
require(ResourceSelection)
library(corrgram)
library(MASS)
library(randomForest)
library(inTrees)
library(pROC)
library(reshape2)
p_load(tidyverse, recipes, rsample, caret, lessR, tidymodels, caret, kknn, e1071, randomForestExplainer)


#Set working directory
setwd("C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets")


# Read in dataset for Regression
yelp <- read.csv("business_Linear_AllDummyNoCensus.csv",na.strings="NULL")
#dim(yelp)

# Remove Cluster field
yelp <- subset(yelp, select = -c(Cluster))

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

###################################
# Variable Importance Plots
###################################

## Term ##

# Feature Importance - using ranger/parsnip
ranger_model <- rand_forest(mode = "classification") %>%
  set_engine("ranger", importance = "impurity") %>%
  fit(stars ~ ., data = yelp[,-c(1)]) #exclude business id

importance_df <- ranger::importance(ranger_model$fit) %>%
  enframe(name = "variable", value = "importance") %>%
  mutate(importance = rescale(importance)) %>%
  mutate_if(is.double, ~round(.x, 4)) %>%
  arrange(-importance)

# Top 20
filtered_df <- filter(importance_df, importance>=0.285)

# Plot
ggplot(filtered_df, aes(x = reorder(variable,importance), y = importance, fill = importance)) + 
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  ylab("Variable Importance") +
  xlab("") +
  ggtitle("Top 20 Variables by Prediction Importance") +
  guides(fill = F) +
  scale_fill_gradient(low = "dodgerblue", high = "dodgerblue4")


#####################
# Train / Test Split
#####################

yelpsample = sample.split(yelp,SplitRatio = 0.75) 
yelp_train =subset(yelp,yelpsample ==TRUE) 
yelp_test=subset(yelp, yelpsample==FALSE) 

#######################################################
# Modeling
# Regular linear regression on training set
# Removed census variables because they are correlated
# Removed target leak variables: review counts
# Include all other variables for future predictions
#######################################################

# Linear Regression model
regmodel=lm(stars~.,data=yelp_train[,-c(1)])
summary(regmodel)

#RMSE
sqrt(mean(regmodel$residuals^2))

# Run model on test data
testmodel = predict(regmodel,newdata=yelp_test[,-c(1)])

# Round down to nearest half a star
output <- cbind(yelp_test, floor(testmodel*2) / 2)

# Create Output File: Examine % accuracy within half a star in Excel
write.csv(output, file = "LinearResults.csv")

