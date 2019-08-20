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

# Read in dataset version 1
raw_df_cat <- read.csv("business.csv",na.strings="NULL")


# Read in dataset version 2
raw_df <- read.csv("business2.csv",na.strings="NULL")
dim(raw_df) #3247  146
str(raw_df)

# Make sure there are no missing values
sum(is.na(raw_df)) 

##### Set up dataframe for clustering: Remove id, binary, and categorical variables #########
cluster_df <- subset(raw_df, select = -c(business_id
                                         ,name
                                         ,address
                                         ,city
                                         ,state_x
                                         ,postal_code
                                         ,latitude
                                         ,longitude
                                         ,is_open
                                         ,cat
                                         ,monday
                                         ,tuesday
                                         ,wednesday
                                         ,thursday
                                         ,friday
                                         ,saturday
                                         ,sunday
                                         ,open_weekdays
                                         ,open_fridays
                                         ,open_weekends
                                         ,weekday_breakfast
                                         ,weekend_breakfast
                                         ,weekday_lunch
                                         ,weekend_lunch
                                         ,weekday_dinner
                                         ,weekend_dinner
                                         ,weekday_late_night
                                         ,weekend_late_night
                                         ,has_parking
                                         ))

dim(cluster_df) #3247  117
str(cluster_df) #stars, review_count, and census data only


attach(cluster_df)

#######################
#### correlation plot##
#######################

require(corrplot)
mcor <- cor(cluster_df)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)
# there's some red in there, red flags
# some variables are negatively correated: need to fix those

# Examine them in small bunches
mcor <- cor(cluster_df[1:20])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)
# problems: poverty_Pct, foreign_Pct, hH_Size

mcor <- cor(cluster_df[21:40])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)
# problems: hH_Food_Stamps, family_Size, hispanic

mcor <- cor(cluster_df[41:60])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

mcor <- cor(cluster_df[61:80])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

mcor <- cor(cluster_df[81:100])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

mcor <- cor(cluster_df[100:117])
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

# Remove offending variables
cluster_df <- subset(cluster_df, select = -c(pop_Density
                                             ,poverty_Pct
                                             ,group_Qtrs_Pct
                                             ,foreign_Pct
                                             ,hH_Size
                                             ,hH_Food_Stamps
                                             ,family_Size
                                             ,hispanic
                                             ,black
                                             ,indian
                                             ,hawaiian
                                             ,other_Race
                                             ,males_0_4
                                             ,males_5_9
                                             ,males_10_14
                                             ,males_15_19
                                             ,males_20_24
                                             ,males_25_29
                                             ,males_30_34
                                             ,males_35_39
                                             ,females_0_4
                                             ,females_5_9
                                             ,females_10_14
                                             ,females_15_19
                                             ,females_20_24
                                             ,females_25_29
                                             ,females_30_34
                                             ,females_35_39
                                             ,females_40_44
                                             ,hu_1970_1979
                                             ,hu_1960_1969
                                             ,hu_1950_1959
                                             ,hu_1940_1949
                                             ,hU_Before_1940
                                             ))

# Retry corrplot
str(cluster_df)

mcor <- cor(cluster_df)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)
# Better. not perfect but definitely improved

#######################
#### Scale Data## ???
#######################
# cluster_df_scale <- as.data.frame(scale(cluster_df))

#######################################################
### check for peaks & valleys (ie) natural clusters ###
#######################################################


# Comare two variables at a time
v1 <- cluster_df[3]
v2 <- cluster_df[4] 

##  Table them
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
# Scaled data: 0.6824335 for comp.2
# Non-scaled: 0.9519887 for comp.2 - very representative of 
# cumulative proportion of variation


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

# Maybe 4-6 clusters

#######################################################
##########  k means with raw data with 5 clusters######
#######################################################

clusterresults <- kmeans(cluster_df,5)
clusterresults$size
rsquare <- clusterresults$betweenss/clusterresults$totss
rsquare
str(clusterresults) # initial r square of 0.8571912


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


##### another way to do the same thing ################

newdf <- as.data.frame(clusterresults$cluster)
pcadf <- as.data.frame(pca$scores)

write.csv(newdf, file = "clusterresults.csv")
write.csv(pcadf, file = "pca.csv")
combdata <- cbind(newdf,pcadf)
head(combdata)

xyplot(Comp.2 ~ Comp.1, combdata, groups = clusterresults$cluster, pch= 20)

################################################################
### Create a dataset with the original data with the cluster info
### This will be useful for creating profiles for the clusters
###############################################################

newdf <- read.csv("clusterresults.csv")

#########################################################
# Profiling: Get averages by cluster for each variable
#########################################################

profiling_df <- subset(raw_df_cat, select = c(is_open
                                              ,Fast_Food
                                              ,temp
                                              ,chain
                                              ,Japanese_dum
                                              ,Pizza_dum
                                              ,Mexican_dum
                                              ,Chinese_dum
                                              ,IceCreamFrozenYogurt_dum
                                              ,Mediterranean_dum
                                              ,Sandwiches_dum
                                              ,Steakhouses_dum
                                              ,AmericanNew_dum
                                              ,DiveBar_dum
                                              ,WineBars_dum
                                              ,Italian_dum
                                              ,Burgers_dum
                                              ,Greek_dum
                                              ,Deli_dum
                                              ,Chicken_dum
                                              ,FishnChips_dum
                                              ,Indian_dum
                                              ,Diners_dum
                                              ,Bars_dum
                                              ,HotDogs_dum
                                              ,Gastropub_dum
                                              ,AmericanTraditional_dum
                                              ,SoulFood_dum
                                              ,BreakfastnBrunch_dum
                                              ,Delis_dum
                                              ,CoffeeTea_dum
                                              ,Cafes_dum
                                              ,CocktailBars_dum
                                              ,Hawaiian_dum
                                              ,SportsBar_dum
                                              ,Korean_dum
                                              ,Vietnamese_dum
                                              ,Sushi_dum
                                              ,Barbeque_dum
                                              ,AsianFusion_dum
                                              ,Salad_dum
                                              ,JuiceBarsSmoothies_dum
                                              ,Donuts_dum
                                              ,Seafood_dum
                                              ,Kosher_dum
                                              ,TexMex_dum
                                              ,Thai_dum
                                              ,Burger_dum
                                              ,Southern_dum
                                              ,Bagels_dum
                                              ,LatinAmerica_dum
                                              ,ModernEuropean_dum
                                              ,FoodCourt_dum
                                              ,French_dum
                                              ,Pretzels_dum
                                              ,African_dum
                                              ,Afghan_dum
                                              ,Cajun_dum
                                              ,Bakeries_dum
                                              ,Vegan_dum
                                              ,Buffets_dum
                                              ,MiddleEastern_dum
                                              ,GlutenFree_dum
                                              ,Polish_dum
                                              ,Caribbean_dum
                                              ,German_dum
                                              ,Vegetarian_dum
                                              ,Spanish_dum
                                              ,Cuban_dum
                                              ,ComfortFood_dum
                                              ,monday
                                              ,tuesday
                                              ,wednesday
                                              ,thursday
                                              ,friday
                                              ,saturday
                                              ,sunday
                                              ,open_weekdays
                                              ,open_fridays
                                              ,open_weekends
                                              ,weekday_breakfast
                                              ,weekend_breakfast
                                              ,weekday_lunch
                                              ,weekend_lunch
                                              ,weekday_dinner
                                              ,weekend_dinner
                                              ,weekday_late_night
                                              ,weekend_late_night
                                              ,has_parking))

combdata <- cbind(cluster_df,newdf,profiling_df)

# dataforgroup <- cbind(cluster_df,newdf,raw_df_cat,raw_df$cat)
# write.csv(dataforgroup, file = "dataforgroup.csv")


require(reshape)
combdata <- rename(combdata, c(clusterresults.cluster="cluster"))
head(combdata)

# Create dataframe of profiling results
profiling_results <- aggregate(combdata,by=list(byvar=combdata$cluster), mean)

# Export profiling results into .csv file
write.csv(profiling_results, file = "profiling_results.csv")

# Get counts of businesses in each cluster
table(combdata$cluster)

# 1    2    3    4    5 
# 536 1236  718  537  220


##############################################################################
# Next, a Hierarchical method is applied to the data with 5 clusters
# which can be done since the dataset is small (less than 10,000 observations)
##############################################################################

numsub.dist = dist(cluster_df)

require(maptree)
hclustmodel <- hclust(dist(cluster_df), method = 'ward.D2') #0.51
#hclustmodel <- hclust(dist(cluster_df), method = 'complete') #0.49
#hclustmodel <- hclust(dist(cluster_df), method = 'single') #0.17
#hclustmodel <- hclust(dist(cluster_df), method = 'average') #0.49

names(hclustmodel)
plot(hclustmodel,xlab="5 Clusters Displayed")
rect.hclust(hclustmodel, k=5, border="red")


cut.5 <- cutree(hclustmodel, k=5)
plot(silhouette(cut.5,numsub.dist),border=NA)
head(cut.5)


########################################
##for hclust how to calculate BSS & TSS
######################################
require(proxy)
numsubmat <- as.matrix(cluster_df)
overallmean <- matrix(apply(numsubmat,2,mean),nrow=1)
overallmean
TSS <- sum(dist(numsubmat,overallmean)^2)
TSS
####################################
#Compute WSS based on 5 clusters
######################################
combcutdata <- cbind(cluster_df,cut.5)
head(combcutdata)

require(reshape)
combcutdata <- rename(combcutdata, c(cut.5="cluster"))
head(combcutdata)

clust1 <- subset(combcutdata, cluster == 1)
clust1 <- as.matrix(clust1,rowby=T)
dim(clust1)
clust1mean <- matrix(apply(clust1,2,mean),nrow=1)
dim(clust1mean)
dis1 <- sum(dist(clust1mean,clust1)^2)

clust2 <- subset(combcutdata, cluster == 2)
clust2 <- as.matrix(clust2,rowby=T)
clust2mean <- matrix(apply(clust2,2,mean),nrow=1)
dis2 <- sum(dist(clust2mean,clust2)^2)

clust3 <- subset(combcutdata, cluster == 3)
clust3 <- as.matrix(clust3,rowby=T)
clust3mean <- matrix(apply(clust3,2,mean),nrow=1)
dis3 <- sum(dist(clust3mean,clust3)^2)

clust4 <- subset(combcutdata, cluster == 4)
clust4 <- as.matrix(clust4,rowby=T)
clust4mean <- matrix(apply(clust4,2,mean),nrow=1)
dis4 <- sum(dist(clust4mean,clust4)^2)

clust5 <- subset(combcutdata, cluster == 5)
clust5 <- as.matrix(clust5,rowby=T)
clust5mean <- matrix(apply(clust5,2,mean),nrow=1)
dis5 <- sum(dist(clust5mean,clust5)^2)

WSS <- sum(dis1,dis2,dis3,dis4,dis5)
WSS

BSS <- TSS - WSS
BSS
## calculating the % of Between SS/ Total SS
rsquare <- BSS/TSS
rsquare


#################################
# PAM method
###############################

clusterresultsPAM <-pam(cluster_df,5)
summary(clusterresultsPAM)
str(clusterresultsPAM$silinfo)
plot(clusterresultsPAM, which.plots=1, main="PAM Clustering: K=5")
#plot(clusterresultsPAM, which.plots=2)

# Average silhouette width per cluster:
#   [1] 0.3408725 0.5896746 0.5366557 0.3439037 0.7064557
# Average silhouette width of total data set:
#   [1] 0.4859221


##########################
## Model based clustering
##########################

library(mclust)
defaultfit <- Mclust(cluster_df)
summary(defaultfit) # display the best model

fit <- Mclust(cluster_df,5)
summary(fit)

# Takes a long time to run
plot(fit,data=cluster_df, what="density") # plot results
#plot(fit,data=cluster_df, what="BIC") # plot results

#compare models
BIC(defaultfit, fit)

dev.off()
dissE <- daisy(cluster_df)
names(dissE)
dE2   <- dissE^2
#sk2   <- silhouette(defaultfit$classification, dE2)
sk2   <- silhouette(fit$classification, dE2)
str(sk2)
plot(sk2,border=NA)

###############################################
## comparison of cluster results  #############
###############################################

#K Means and PAM
clstat <- cluster.stats(numsub.dist, clusterresults$cluster, clusterresultsPAM$cluster)
names(clstat)
clstat$corrected.rand #0.3579303
##corrected or adjusted rand index lies between 0 & 1
## perfect match between 2 clustering methods means 1, no match means 0
## any number in between represents 'kind of' % of matched pairs 

#K Means and Hierarchical
clstat <- cluster.stats(numsub.dist, clusterresults$cluster, cut.5)
clstat$corrected.rand #0.43902

# PAM and Hierarchical
clstat <- cluster.stats(numsub.dist, clusterresultsPAM$cluster, cut.5)
clstat$corrected.rand



############################################################

## Add cluster results as variable to dataframe
#raw_df_cat$cluster <- clusterresults$cluster

