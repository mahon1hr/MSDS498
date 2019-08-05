# Libraries
require(cluster)
require(useful)
require(Hmisc)
require(plot3D)
library(HSAUR)
library(MVA)
library(HSAUR2)
library(fpc)
library(mclust)
library(lattice)
library(car)
library(digest)
require(corrplot)
library(latticeExtra)
library(dplyr)
library(stargazer)
library(pROC)
p_load(tidyverse, recipes, rsample, caret, lessR, tidymodels, caret, kknn, e1071)

#Set working directory
setwd("C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets")

# Read in data
raw_df <- read.csv("business.csv",na.strings="NULL")
dim(raw_df) #3247  221
str(raw_df)
View(raw_df)

# Define output path for html files;
out.path <- 'C:/Users/cscanlon/Desktop/SQL/NW/12-CAPSTONE/Project/Datasets/';

###### Basic EDA ########

# Generate Summary Statistics table 
file.name <- 'SummaryTable.html';
stargazer(raw_df[9:221], type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 1.1: Summary Statistics for Numerical Variables'),
          align=TRUE, digits=0, digits.extra=2, initial.zero=TRUE, median=TRUE)

#lessR quick graphs for Numerical/Discrete Variables 
lessR::Histogram(data = raw_df[9:243])

# Make sure there are no missing values
sum(is.na(raw_df)) 

# For each zip code, calculate the % total and sort in descending order
grouped_df <- raw_df %>%
  group_by(postal_code) %>%
  summarise(perc_tot = n()/3533) %>%
  arrange(desc(perc_tot))

# Open vs. Not Open
raw_df %>%
  group_by(is_open) %>%
  summarise(perc_tot = n()/3533) %>%
  arrange(desc(perc_tot))

# Stars
raw_df %>%
  group_by(stars) %>%
  summarise(perc_tot = n()/3533) %>%
  arrange(desc(stars))

########### Set up dataframe for clustering #########
cluster_df <- subset(raw_df, select = -c(business_id
                                         ,name
                                         ,address
                                         ,city
                                         ,state_x
                                         ,postal_code
                                         ,latitude
                                         ,longitude
                                         ))

dim(cluster_df) #3247  213

attach(cluster_df)

#######################
#### correlation plot##
#######################
require(corrplot)
mcor <- cor(cluster_df)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

#######################
#### Scale Data## ???
#######################
#cluster_df <- as.data.frame(scale(cluster_df))

#######################################################
### check for peaks & valleys (ie) natural clusters ###
#######################################################

# Comare two variables at a time
v1 <- cluster_df[200]
v2 <- cluster_df[201] 

##  Tablem them
z <- table(v1, v2)
z

##  Plot as a 3D histogram:
hist3D(z=z, border="black")

# Received error: need to investigate further

#######################################
############### PCA Plots ##############
######################################
dev.off()
pca <-princomp(cluster_df)
plot(pca$scores[,1],pca$scores[,2])

names(pca)
str(pca)
summary(pca)
head(pca$scores)

sort(pca$scores[,1])

#Check other end
sort(pca$scores[,1], decreasing = TRUE)

##########################################################

##  Create cuts:
pcadf <- as.data.frame(pca$scores)
pca1 <- cut(pcadf$Comp.1, 10)
pca2 <- cut(pcadf$Comp.2, 10)

head(pcadf)

##  Calculate joint counts at cut levels:
z <- table(pca1, pca2)

##  Plot as a 3D histogram:
hist3D(z=z, border="black")

###################################################
### Create a 'scree' plot to determine the num of clusters
#####################################################

dev.off()
wssplot <- function(cluster_df, nc=15, seed=1234) {
  wss <- (nrow(cluster_df)-1)*sum(apply(cluster_df,2,var))
  for (i in 2:nc) {
    set.seed(seed)
    wss[i] <- sum(kmeans(cluster_df, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")} 

wssplot(cluster_df)

# Maybe 4 - 6 clusters

#######################################################
##########  k means with raw data with 5 clusters######
#######################################################

clusterresults <- kmeans(cluster_df,5)
clusterresults$size
rsquare <- clusterresults$betweenss/clusterresults$totss
rsquare
str(clusterresults)


#R automatically plots using PCA1 and PCA2 due to # of dimensions
plot(clusterresults, data=cluster_df)

# Silhouette Plot

#dev.off()
dissE <- daisy(cluster_df)
names(dissE)
dE2   <- dissE^2
sk2   <- silhouette(clusterresults$cluster, dE2)
str(sk2)

plot(sk2, border=NA, main='Silhouette Plot', col=blues9)




