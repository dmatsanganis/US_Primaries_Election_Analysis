# US_Primaries_Election_Analysis

################## Start: Load required libraries ##############################

library(dplyr)
library(readxl)
library(tidyr)
library(tidyverse)
library(car)
library(glmnet)
library(DescTools)
library(aod)
library(vcdExtra)
library(ggplot2)
library(gridExtra)
library(corrplot) 
library(scales)
library(rattle)
library(rpart)
library(rpart.plot)
library(class)
library(e1071)
library(caret)
library(ggplot2)
library(mclust)
library(tree)
library(ROCR)
library(pROC)
library(pgmm)
library(cluster)
library(NbClust)
library(jpeg)
library(fpc)
library(stats)
library(devtools)
library(factoextra)
library(dendextend)
library(corrplot)

################### END: Load required libraries ###############################


####################################################
#########  Dataset Preparation & Cleaning  #########
####################################################

# Import the dataset and create 3 dataframes.
county_facts = read_excel("stat BA II project I.xlsx", sheet = "county_facts")
votes = read_excel("stat BA II project I.xlsx", sheet = "votes")
dictionary = read_excel("stat BA II project I.xlsx", sheet = "dictionary")

# Preview the dataframes.
# View(county_facts)
# View(votes)
# View(dictionary)

# Remove the Democrats, since we care about the Republicans elections only.
# We do so with the assistance of dplyr's filter.
votes = filter(votes, party == "Republican")


###################### New Hampshire State Case ################################

# New Hampshire State has empty cells, regarding it's FIPS code.
# We should fill these cells, since there are important for the merging later, 
# which will be performed indexed by the FIPS code.

# For starters, we need to find the columns with the missing values 
# and then sort them.
na_counter = which(colSums(is.na(votes)) > 0)
sort(colSums(sapply(votes[na_counter], is.na))) 

# Get the missing data from the other Excel sheet and create a temp table.
temp = subset(county_facts, county_facts$state_abbreviation == "NH")[1:3]
temp = separate(temp, col = area_name, into = c("state","county_label"), sep = " ")

# Import the missing FIPS codes.
votes$fips[is.na(votes$fips)] = temp$fips[match(votes$county, temp$state)][which(is.na(votes$fips))]

# Check again for missing values - returns 0.
na_counter = which(colSums(is.na(votes)) > 0)
na_counter

# Remove `temp` and `na_counter`, no further use.
rm(temp, na_counter)

###################### End: New Hampshire State Case ###########################

# Now let's prepare the county_facts prior to the merge.
# View(county_facts)
# We begin with the removal of states rows 
# - the ones with empty 'state_abbreviation' column.
county_facts = filter(county_facts, state_abbreviation != "")

# Now, we remove the " County" substring, to match the other dataframe notation
# and all the fields now referred to counties.
county_facts$area_name  = str_replace(county_facts$area_name, " County", "")

# Rename also the column's name to 'county' for better understanding purposes.
colnames(county_facts)[2] = "county"

# We encounter some different notations for some counties 
# (WY, VT, RI, ND, MA, ME, KS, IL, CT, AK).
# After research we found out that these codes referred to towns/cities or 
# smaller portions of the counties (with the exception of the Alaska, Kansas and
# Wyoming, where we could not relate the given data with the counties level).
# Therefore, for the remaining (VT, RI, ND, MA, ME, IL, CT) we could just drop 
# them, but since we aim to be an analyst we decide to research and create a CSV
# that relates the towns/cities to counties, then we select, groupby and 
# summarize the data based on the FIPS code (which was provided by the 
# county_facts dataset) for each of the above-mentioned states. 
# Then, we merge (vertical - rbind) the correct FIPS codes data with the votes 
# dataset, prior to the creation of the response variable 
# (Trump's votes over 50%). 
# Below we describe this procedure for each of the counties.

# We need to point out that the ME and ND, contains only votes for the Democrats
# and we do not need to find the correct FIPS, since it will be dropped (we only
# care for Republicans).

# Furthermore, the AK contains instead of towns/cities that can be related to 
# counties, State House Districts. Thus, we need also to drop those since a 
# State House District is referred to more than one counties and that means that 
# we cannot sum the votes and have proper results. A similar case occurs with 
# the KS state where there are Congressional Districts that again cover more 
# than one counties (e.g. Anderson, KS belongs both to Congressional District 1
# and to Congressional District 2). Therefore, again we cannot extract valid 
# vote data from it. Finally, the WY, contains votes data for counties on pairs
# and again we cannot find out to which county the data refers so we decide to 
# drop WY too.

# In conclusion, from the FIPS-issue counties 
# WY, VT, RI, ND, MA, ME, KS, IL, CT, AK), we decide to drop the ones that either 
# do not contain Republicans votes or we cannot extract valid data from those.
# Thus, we drop ME, ND, AK, KS, WY for the above reasons and the remaining are
# the following:  VT, RI, MA, IL, CT.

##################### Massachusetts FIPS codes issue ########################### 

df_MA_votes = filter(votes, state_abbreviation == "MA")
# View(df_MA_votes)

# Correct FIPS codes to counties repository, used as reference to relate data.
df_MA = filter(county_facts, state_abbreviation == "MA")
df_MA = select(df_MA, c('fips', 'county'))
# View(df_MA)

# Towns/Cities to Counties.
df_MA_cities_to_counties = read.csv("MA.csv", header = T)
# View(df_MA_cities_to_counties)

# Merge the towns/cities with the counties. 
# Now there is a relation, we can add the correct FIPS codes.
df_MA_cities_to_counties = merge(df_MA_cities_to_counties, df_MA_votes, by = "county")

# Delete the wrong FIPS notation.
df_MA_cities_to_counties = select(df_MA_cities_to_counties, -fips)

# Rename with proper names.
colnames(df_MA_cities_to_counties)[1] = "town/city"
colnames(df_MA_cities_to_counties)[2] = "county"

# Get the FIPS codes from the reference one.
df_MA_cities_to_counties = merge(df_MA_cities_to_counties, df_MA, by = "county")

# Format the dataframe, a procedure which we will apply to the general one, in 
# order to create the response variable.
df_MA_cities_to_counties = select(df_MA_cities_to_counties, c('fips', 'candidate', 'votes')) 
df_MA_cities_to_counties = group_by(df_MA_cities_to_counties, fips, candidate)
df_MA_cities_to_counties = summarize(df_MA_cities_to_counties, Total_Votes = sum(votes))

# We will merge this one with the general votes one, after we follow the same
# procedure for it, in order for the dataframes to be in the same format.
df_MA = spread(df_MA_cities_to_counties, candidate, Total_Votes)

# Remove the dataframes with no further use - simplify the data, keep only the
# df_MA.
rm(df_MA_cities_to_counties, df_MA_votes)

# Preview the dataframe.
# View(df_MA)

################### END: Massachusetts FIPS codes issue ######################## 

####################### Illinois FIPS codes issue ############################## 

# Regarding the Illinois FIPS codes, there are only observed issues with two 
# cities. These cities are `Cook Suburbs` and `Chicago`, that both belong 
# to `Cook County` that has the FIPS code `17031`. Therefore, prior to the 
# format and the groupby of the dataframe, we should assign this FIPS code to 
# both cities of the votes dataframe.

# Assign the FIPS code to both the counties at the original votes dataframe.
votes$fips[votes$county %in% c("Cook Suburbs", "Chicago")] = 17031

# No need for further merge regarding the Illinois case.

##################### END: Illinois FIPS codes issue ########################### 

###################### Connecticut FIPS codes issue ############################ 

# With the Connecticut there are cities/towns that we need to relate first with 
# counties then with the FIPS codes and the - like we did with MA, we will 
# format the dataframe prior to that latest merge before the creation of the  
# response column (Trump over 50% of the Republicans votes).
# We will follow a procedure like the one followed for the MA's case.

df_CT_votes = filter(votes, state_abbreviation == "CT")
# View(df_CT_votes)

# Correct FIPS codes to counties repository, used as reference to relate data.
df_CT = filter(county_facts, state_abbreviation == "CT")
df_CT = select(df_CT, c('fips', 'county'))
# View(df_CT)

# Towns/Cities to Counties.
df_CT_cities_to_counties = read.csv("CT.csv", header = T)
# View(df_CT_cities_to_counties)

# Merge the towns/cities with the counties. 
# Now there is a relation, we can add the correct FIPS codes.
df_CT_cities_to_counties = merge(df_CT_cities_to_counties, df_CT_votes, by = "county")

# Delete the wrong FIPS notation.
df_CT_cities_to_counties = select(df_CT_cities_to_counties, -fips)

# Rename with proper names.
colnames(df_CT_cities_to_counties)[1] = "town/city"
colnames(df_CT_cities_to_counties)[2] = "county"

# Get the FIPS codes from the reference one.
df_CT_cities_to_counties = merge(df_CT_cities_to_counties, df_CT, by = "county")

# Format the dataframe, a procedure which we will apply to the general one, in 
# order to create the response variable.
df_CT_cities_to_counties = select(df_CT_cities_to_counties, c('fips', 'candidate', 'votes')) 
df_CT_cities_to_counties = group_by(df_CT_cities_to_counties, fips, candidate)
df_CT_cities_to_counties = summarize(df_CT_cities_to_counties, Total_Votes = sum(votes))

# We will merge this one with the general votes one, after we follow the same
# procedure for it, in order for the dataframes to be in the same format.
df_CT = spread(df_CT_cities_to_counties, candidate, Total_Votes)

# Remove the dataframes with no further use - simplify the data, keep only the
# df_CT.
rm(df_CT_cities_to_counties, df_CT_votes)

# Preview the dataframe.
# View(df_CT)

#################### END: Connecticut FIPS codes issue ######################### 

###################### Rhode Island FIPS codes issue ########################### 

# Rhode Island (RI) state has 39 towns/cities that we need to relate them to the 
# five counties of the state, we can do so with the `RI.csv` that contains the 
# relation data between the cities and the counties and have been created 
# through the web.
# Again we will follow a procedure like the two states above to create a final
# dataframe, which will be ready (in the same format) with the general one and
# will be merge prior to the creation of the response procedure.
# We need to point out also, that in RI's case there was 2 entries refereed to 
# mail ballots, `Mail Ballots C.D. 1` and `Mail Ballots C.D. 2`, after our 
# research these kind of data cannot be assigned to a specif county or even city
# thus we decide to drop them and continue with the votes of the 39 RI's cities.

df_RI_votes = filter(votes, state_abbreviation == "RI")
# View(df_RI_votes)

# Correct FIPS codes to counties repository, used as reference to relate data.
df_RI = filter(county_facts, state_abbreviation == "RI")
df_RI = select(df_RI, c('fips', 'county'))
# View(df_RI)

# Towns/Cities to Counties.
df_RI_cities_to_counties = read.csv("RI.csv", header = T)
# View(df_RI_cities_to_counties)

# Merge the towns/cities with the counties. 
# Now there is a relation, we can add the correct FIPS codes.
df_RI_cities_to_counties = merge(df_RI_cities_to_counties, df_RI_votes, by = "county")

# Delete the wrong FIPS notation.
df_RI_cities_to_counties = select(df_RI_cities_to_counties, -fips)

# Rename with proper names.
colnames(df_RI_cities_to_counties)[1] = "town/city"
colnames(df_RI_cities_to_counties)[2] = "county"

# Get the FIPS codes from the reference one.
df_RI_cities_to_counties = merge(df_RI_cities_to_counties, df_RI, by = "county")

# Format the dataframe, a procedure which we will apply to the general one, in 
# order to create the response variable.
df_RI_cities_to_counties = select(df_RI_cities_to_counties, c('fips', 'candidate', 'votes')) 
df_RI_cities_to_counties = group_by(df_RI_cities_to_counties, fips, candidate)
df_RI_cities_to_counties = summarize(df_RI_cities_to_counties, Total_Votes = sum(votes))

# We will merge this one with the general votes one, after we follow the same
# procedure for it, in order for the dataframes to be in the same format.
df_RI = spread(df_RI_cities_to_counties, candidate, Total_Votes)

# Remove the dataframes with no further use - simplify the data, keep only the
# df_RI.
rm(df_RI_cities_to_counties, df_RI_votes)

# Preview the dataframe.
# View(df_RI)

################### END: Rhode Island FIPS codes issue #########################

######################## Vermont FIPS codes issue ############################## 

# Regarding Vermont (VT) contains 246 incorporated towns and cities. 
# 9 of them are cities and 237 are towns. Thus we collect the relation data to
# a CSV-file, `VT.csv`, and we followed the above procedure to create the final 
# dataframe, df_VT.

df_VT_votes = filter(votes, state_abbreviation == "VT")
# View(df_VT_votes)

# Correct FIPS codes to counties repository, used as reference to relate data.
df_VT = filter(county_facts, state_abbreviation == "VT")
df_VT = select(df_VT, c('fips', 'county'))
# View(df_VT)

# Towns/Cities to Counties.
df_VT_cities_to_counties = read.csv("VT.csv", header = T)
# View(df_VT_cities_to_counties)

# Merge the towns/cities with the counties. 
# Now there is a relation, we can add the correct FIPS codes.
df_VT_cities_to_counties = merge(df_VT_cities_to_counties, df_VT_votes, by = "county")

# Delete the wrong FIPS notation.
df_VT_cities_to_counties = select(df_VT_cities_to_counties, -fips)

# Rename with proper names.
colnames(df_VT_cities_to_counties)[1] = "town/city"
colnames(df_VT_cities_to_counties)[2] = "county"

# Get the FIPS codes from the reference one.
df_VT_cities_to_counties = merge(df_VT_cities_to_counties, df_VT, by = "county")

# Format the dataframe, a procedure which we will apply to the general one, in 
# order to create the response variable.
df_VT_cities_to_counties = select(df_VT_cities_to_counties, c('fips', 'candidate', 'votes')) 
df_VT_cities_to_counties = group_by(df_VT_cities_to_counties, fips, candidate)
df_VT_cities_to_counties = summarize(df_VT_cities_to_counties, Total_Votes = sum(votes))

# We will merge this one with the general votes one, after we follow the same
# procedure for it, in order for the dataframes to be in the same format.
df_VT = spread(df_VT_cities_to_counties, candidate, Total_Votes)

# Remove the dataframes with no further use - simplify the data, keep only the
# df_VT.
rm(df_VT_cities_to_counties, df_VT_votes)

# Preview the dataframe.
# View(df_VT)

##################### END: Vermont FIPS codes issue ############################

############## us_elections_2016 - General votes dataframe #####################

# Now we can move forward and create the 'us_elections_2016' dataframe. 
us_elections_2016 = votes

# Perform the same procedure for our main votes dataframe.
us_elections_2016 = select(us_elections_2016, c('fips', 'candidate', 'votes')) 
us_elections_2016 = group_by(us_elections_2016, fips, candidate)
us_elections_2016 = summarize(us_elections_2016, Total_Votes = sum(votes))
us_elections_2016 = spread(us_elections_2016, candidate, Total_Votes)

# Drop the wrong FIPS codes - since they cannot be assign with the `county_facts`
# dataframe. Thus drop the over 5-digit 'FIPS' codes (there are not FIPS codes).
# As we said earlier we will input the proper FIPS codes from the above created 
# dataframes for the states that had an issue.
us_elections_2016 = filter(us_elections_2016, fips <= 99999)

# Preview the dataframe.
# View(us_elections_2016)

# Add the Massachusetts (MA) correct FIPS codes. 
us_elections_2016 = rbind(us_elections_2016, df_MA)

# Add the Connecticut (CT) correct FIPS codes. 
us_elections_2016 = rbind(us_elections_2016, df_CT)

# Add the Rhode Island (RI) correct FIPS codes. 
us_elections_2016 = rbind(us_elections_2016, df_RI)

# Add the Vermont (VT) correct FIPS codes. 
us_elections_2016 = rbind(us_elections_2016, df_VT)

# After the merge, we do not need anymore the separate dataframes, thus we 
# drop them.
rm(df_MA, df_CT, df_RI, df_VT, votes)

# Replace the NAs with 0, in order to make the calculation
# of the total Republicans votes.
us_elections_2016 = mutate(us_elections_2016, across(everything(), ~replace_na(.x, 0)))

# Since the size of the data frame is large and has lots of columns,
# the mutate operation needs a while to complete.

# Preview the dataframe.
# View(us_elections_2016)

# Add all Republicans' Candidates votes.
# 1: FIPS code.     2: Ben Carson.  3: Carly Fiorina.  4:Chris Christie. 
# 5: Donald Trump.  6: Jeb Bush.    7: John Kasich.    8: Marco Rubio. 
# 9: Mike Huckabee. 10: Rand Paul. 11: Rick Santorum. 12:Ted Cruz.

# Create a new column per FIPS code (per county), republicans_votes.
us_elections_2016$republicans_votes = rowSums(us_elections_2016[2:12]) 

# Create our response variable by dividing the total votes by 2 and then compare
# it Trump's votes for each county. 
# 0: Trump < 50%. 1: Trump > 50% of the Republicans votes.
us_elections_2016$trump_over_perc50 = ifelse(us_elections_2016$`Donald Trump` >
                                               us_elections_2016$`republicans_votes`/2, 1, 0)


# Drop the unnecessary columns: the 11 candidates and the republicans_votes.
# Keep only the FIPS codes for the later merge and the response column.
us_elections_2016 = select(us_elections_2016, c('fips', 'trump_over_perc50')) 

# e.g. 4027 1
#      5001 0
#      5003 0

# Output the data frame to a CSV-file, for testing purposes.
# write.csv(us_elections_2016, file = "us_elections_2016.csv", row.names = FALSE)

############### END: us_elections_2016 - General votes dataframe ###############

############## Merge the two dataframes to create the final one ################

# Now we need to merge the the `county_facts` with the `us_elections_2016`, 
# based on the FIPS code.
us_elections_2016 = merge(county_facts, us_elections_2016, by = "fips")

# We found out that some states that exist in the `county_facts` do not exist
# in the given initial `votes` dataset. These states are: Colorado (8XXX), 
# District of Columbia (11XXX), Minnesota (27XXX), and North Dakota (38XXX). 
# Of course the ones we mention earlier are missing from the dataframe (AK, ME
# ND, KS, and WY).

# Now we can also remove the `county_facts` dataframe 
# - since we do not need it anymore.
rm(county_facts)

# Output the data frame to a CSV-file, for testing purposes.
# write.csv(us_elections_2016, file = "us_elections_2016FINAL.csv", row.names = FALSE)

# File Checkpoint - Read in the data.
us_elections_2016 = read.csv("us_elections_2016FINAL.csv", header = TRUE)

# Remove non-covariate columns (fips, county).
us_elections_2016 = us_elections_2016[, -c(1:2)]

# Make the response variable a factor (since it is binary) from int.
us_elections_2016$trump_over_perc50 = as.factor(us_elections_2016$trump_over_perc50)

# Preview the dataframe to check that the correct columns became factors.
str(us_elections_2016)

# With the assistance of dplyr move the response variable 'trump_over_perc50' 
# to the end of the dataframe - last column of the dataframe.
us_elections_2016 = relocate(us_elections_2016, trump_over_perc50, .after = last_col())

# Validate the changes.
# View(us_elections_2016)

# Output the data frame to a CSV-file, for testing purposes.
# write.csv(us_elections_2016, file = "us_elections_2016FINAL.csv", row.names = FALSE)

########### END: Merge the two dataframes to create the final one ##############

######################## START: Classification Part I ##########################

######################## START: Data Partitioning ##############################

# File Checkpoint - Read in the data.
us_elections_2016 = read.csv("us_elections_2016FINAL.csv", header = TRUE)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Create a data partition for the response variable 'trump_over_perc50' 
# with a 70/30 split. The 'list = FALSE' argument ensures that the 
# output is a vector rather than a list.
trainIndex = createDataPartition(us_elections_2016$trump_over_perc50,
                                 p = 0.7, list = FALSE)

# Subset the original data into a training set and a test set 
# using the indices from the data partition.
train_data = us_elections_2016[trainIndex, ]
test_data = us_elections_2016[-trainIndex, ]

# Create a train control object for use in model training with repeated K-fold
# cross-validation. The 'number' parameter sets the number of folds to use 
# in each iteration of cross-validation, while the 'repeats' parameter sets the 
# number of times to repeat the cross-validation process.
trainControl = trainControl(method = "repeatedcv", number = 10, repeats = 20)

# Reasoning:
# If we simply split your data into a training set and a test set, we might get
# different results each time we run our model, depending on which data points 
# end up in the test set. This can make it difficult 
# to know whether the model is actually performing well or not.

# To be more precise, there is always a chance that the results you get are due 
# to chance variations in the data. Using trainControl helps us to set up a 
# standardized and robust way to train and evaluate the models. This way we can
# be more confident in the accuracy of the results and avoid overfitting of 
# the models to the training data. To sum up, this way we ensure that the models
# are being trained and evaluated in a consistent and robust way.

######################## END: Data Partitioning ################################

################# START: Logistic Regression - Classification ##################

# Construct the full model.
model_full = glm(trump_over_perc50 ~ ., data = train_data, family = binomial())  

# Preview the model's summary.
summary(model_full) 

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Stepwise procedure based on the AIC (since we have predictions) criterion. 
model_aic = step(model_full, direction = "both")

# Preview the model's summary.
summary(model_aic) 
# AIC 980.86

# With the vif() function from the car package we calculate the Variance 
# Inflation Factor (VIF) for each covariate, while we round the VIF values
# to one decimal for our convenient.
round(vif(model_aic), 1)

# Multicollinearity issue!

# Remove the covariates with highest VIF value each time and update the model,
# until all VIF values are under the '10' threshold.
model_aic = update(model_aic, trump_over_perc50 ~ . - PST040210)
model_aic = update(model_aic, trump_over_perc50 ~ . - BZA010213)
model_aic = update(model_aic, trump_over_perc50 ~ . - POP010210)
model_aic = update(model_aic, trump_over_perc50 ~ . - BZA110213)
model_aic = update(model_aic, trump_over_perc50 ~ . - POP645213)
model_aic = update(model_aic, trump_over_perc50 ~ . - INC910213)
# model_aic = update(model_aic, trump_over_perc50 ~ .- state_abbreviation)

summary(model_aic)
# No multicollinearity issue! 
# Everything are under the threshold of 10.

# length(model_aic$coefficients) # 18

# Wald test (aod library).
wald.test(b = coef(model_aic), Sigma = vcov(model_aic), Terms = 2:18)

# Chi-squared test:
# X2 = 128.7, df = 17, P(> X2) = 0.0

# With the above results of the Wald test resulted in a chi-squared value of 
# 128.7 with 17 degrees of freedom, and a p-value of 0, indicates that at least
# one of the coefficients in the model is significantly different from zero.

# The results of a Wald test does not necessarily mean that all coefficients in 
# the model are significant.

# Likelihood Ratio Test (LRT).

# The difference in deviances between the null model (only the intercept) 
# and the current model (aic model).
# This value represents the amount of unexplained variability that 
# the full model reduces compared to the null model.
with(model_aic, null.deviance - deviance)
# 1771.333

# Calculates the degrees of freedom for the null model minus the degrees of 
# freedom for the full model.
with(model_aic, df.null - df.residual)
# 63

# Calculates the p-value for the goodness of fit of the full model. 
# It is calculated as the probability of observing a deviance as large or 
# larger than the one obtained in the full model, 
# given the number of degrees of freedom for the residual.
with(model_aic, pchisq(deviance, df.residual, lower.tail = FALSE)) 
# 1

# Calculates the p-value for the overall model fit, comparing the null model 
# with the full model. It is calculated as the probability of observing a 
# difference in deviances as large or larger than the one obtained in the 
# full model, given the number of degrees of freedom.
with(model_aic, pchisq(null.deviance - deviance, 
                       df.null - df.residual, lower.tail = FALSE))
# 0

# In our case, the p-value for the goodness of fit of the full model (1 pchisq) 
# is 1, indicating a good fit, and the p-value for the overall model fit is very
# close to 0,indicating that the full model is significantly better 
# than the null model.

# We use the PseudoR2 function of the DescTools package to calculate the 
# pseudo-R2 for a logistic regression model. 
# All 3 variation of R2 that were discussed in the class.
PseudoR2(model_aic, 'all')

# McFadden      CoxSnell        Nagelkerke 
# 0.6750022     0.5996507       0.8077688  

# Exponentiation of the coefficients and interpret them as odds-ratios.
exp(coef(model_aic))
exp(cbind(OR = coef(model_aic), confint(model_aic)))

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Train the log model.
model_log = train(trump_over_perc50 ~ AGE295214 + RHI525214 + RHI625214 + 
                  POP815213 + EDU635213 + EDU685213 + VET605213 + 
                  HSG096213 + INC110213 + PVY020213 + NES010213 + 
                  SBO315207 + MAN450207 + AFN120207 + LND110210 + 
                  state_abbreviation, 
                  data = train_data,
                  method ="glm", 
                  family = "binomial",
                  trControl = trainControl)

# Output the metrics.
model_log$results

# RMSE       Rsquared    MAE     
# 0.2891628  0.6548606   0.1597477

# Output the metrics.
model_log

# Assessing the predictive ability of the model (ofc. exclude the response).
fitted.results = predict(model_log, test_data [,-53], type = 'raw')
fitted.results = ifelse(fitted.results > 0.5, 1, 0)

# Accuracy check.
# accuracy = 1 - mean(fitted.results != test_data$trump_over_perc50)
# accuracy
# 0.8853

# Create the 2x2 confusion matrix.
table(test_data$trump_over_perc50, fitted.results)

# Logistic model metrics.
log_metric = confusionMatrix(as.factor(fitted.results), 
                             as.factor(test_data$trump_over_perc50),
                             positive = "1")
log_metric
# 0.8853         

# Prepare the ROC curve (TPR and FPR from predictions).
log_pred = prediction(fitted.results, test_data$trump_over_perc50)
log_roc = performance(log_pred, "tpr", "fpr")

# Output the ROC curve.
plot(log_roc, col = "steelblue3", main = "ROC Curve - Logistic Regression Model", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label.
abline(0, 1, lty = 2)  # Add diagonal reference line.
# legend("bottomright", legend = c("Logistic Regression Model"),
#        col = c("steelblue3"), lty = 1, cex = 1.2) 

################## END: Logistic Regression - Classification ###################

################### START: Decision Tree - Classification ######################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Create the Decision Tree model with the train data.
dt_model = rpart(trump_over_perc50 ~ ., data = train_data, method = 'class')

# Output the findings.
printcp(dt_model)
rpart.rules(dt_model, style = "tall")

# Output the covariates importance for the Decision Tree model.
dt_model$variable.importance

# Output complexity info regarding the fitting model.
# dt_model$cptable

# Predict the unseen data (without the response).
dt_class = predict(dt_model, test_data[, -53], type = 'class')
dt_class

# Create the 2x2 confusion matrix.
table(test_data$trump_over_perc50, dt_class)

# Decision Tree model metrics.
dt_metrics = confusionMatrix(dt_class, 
                             as.factor(test_data$trump_over_perc50),
                             positive = "1")

# Output the results.
dt_metrics
# 87.44

# Further evaluation: accuracy of the model.  
# sum(diag(table(test_data$trump_over_perc50, dt_class))) /
#   sum(table(test_data$trump_over_perc50, dt_class))

# Alternative way with tree library: 
# dt_model2 = tree(as.factor(trump_over_perc50) ~ ., data = train_data)
# dt_class2 = predict(dt_model2, test_data, type = 'class')
# table(test_data$trump_over_perc50, dt_class2)

# Prepare the ROC curve (TPR and FPR from predictions).
dt_pred = prediction(as.integer(dt_class)-1, test_data$trump_over_perc50)
dt_roc = performance(dt_pred, "tpr", "fpr")

# Output the ROC curve.
plot(dt_roc, col = "steelblue3", main = "ROC Curve - Decision Tree Model", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label.
abline(0, 1, lty = 2)  # Add diagonal reference line.

# Output the model.
fancyRpartPlot(dt_model, main = "Decision Tree Model", sub = " ", type = 2)

################### END: Decision Tree - Classification ########################

######################## START: Naive Bayes Model ##############################

# Construct the Naive Bayes Model.
nb_model = naiveBayes(y = train_data$trump_over_perc50, x = train_data)

# Predict the unseen data.
nb_class = predict(nb_model, newdata = test_data)

# Preview the results.
# nb_class

# Create the 2x2 confusion matrix.
table(test_data$trump_over_perc50, nb_class)

# Naive Bayes Classifier metrics.
nb_metrics = confusionMatrix(nb_class,
                             as.factor(test_data$trump_over_perc50),
                             positive = "1")

# Output the results.
nb_metrics
# 0.971      

# Further evaluation: accuracy of the model.  
# sum(diag(table(test_data$trump_over_perc50, nb_class))) /
#   sum(table(test_data$trump_over_perc50, nb_class))

# Prepare the ROC curve (TPR and FPR from predictions).
nb_pred = prediction(as.integer(nb_class)-1, test_data$trump_over_perc50)
nb_roc = performance(nb_pred, "tpr", "fpr")

# Output the ROC curve.
plot(nb_roc, col = "steelblue3", main = "ROC Curve - Naive Bayes", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label.
abline(0, 1, lty = 2)  # Add diagonal reference line.

# !!!!!!!!!!!!! Limitations !!!!!!!!!!!!! 

########################## END: Naive Bayes Model ##############################

########################## START: SVM Model ####################################

# First we need to check the datasets' dimensions. 
dim(train_data)
dim(test_data)

# We scale the data prior to the application of the SVM, to ensure that all 
# predictors are on the same scale. This way, we are led to better model 
# performance and faster convergence times. 
# train_scale = scale(train_data)

# Alternative: With the svm() function from the e1071 package which is used 
# specifically for fitting SVM models. We are using svm() to fit an SVM model 
# with a linear kernel. However, unlike train(), svm() does not allow us to 
# specify a resampling method. For this reason, we choose to fit our SVM model
# through the above way. The results are similar as we could expect, but the 
# prior method offers us more reliable results.
# For this reason we choose the caret library option, which provide us the 
# flexibility to resample through CV and trainControl.
svm_model = svm(trump_over_perc50 ~ ., data = train_data)
# svm_model = svm(as.factor(trump_over_perc50) ~ ., data = train_scale)

# Create and train the SVM model.
# With the assistance of the caret package is used for training a variety of
# machine learning models, including SVM. Thus, we are using train() to fit a 
# SVM model with a linear kernel (method = "svmLinear"). 
# This way also allows us to specify a number of other arguments, including 
# the resampling method (cross-validation with trainControl argument) to use.
svm_model = train(as.factor(trump_over_perc50) ~., 
                  data = train_data,
                  method = "svmLinear",
                  trControl = trainControl)


# Predict the unseen data (excluding the response variable).
svm_class = predict(svm_model, test_data[,-53])

# Preview the results.
# svm_class

# Create the 2x2 confusion matrix.
table(test_data$trump_over_perc50, svm_class)

# SVM model metrics (Confusion matrix).
svm_metric = confusionMatrix(svm_class, 
                             as.factor(test_data$trump_over_perc50),
                             positive = "1")
svm_metric 
# 88.65

# Further evaluation: accuracy of the model.  
# sum(diag(table(test_data$trump_over_perc50, svm_class))) /
#   sum(table(test_data$trump_over_perc50, svm_class))

# Prepare the ROC curve (TPR and FPR from predictions).
svm_pred = prediction(as.integer(svm_class)-1, test_data$trump_over_perc50)
svm_rocs = performance(svm_pred, "tpr", "fpr")

# Output the ROC curve.
plot(svm_rocs, col = "steelblue3", main = "ROC Curve - SVM Model", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label.
abline(0, 1, lty = 2)  # Add diagonal reference line.

############################### END: SVM Model #################################

########################## START: KNN Model ####################################

# First we need to check the datasets' dimensions. 
dim(train_data)
dim(test_data)

# Now since the dimensions are equal we are moving forward to scale
# and standardize the data. To be more precise, since KNN is a distance-based 
# algorithm and the distance metric used by KNN is affected by the scale of 
# the input features. Thus, when features are not on the same scale, 
# the feature with the largest range will dominate the distance metric.
# Therefore, by scaling the data prior to implementing KNN, we ensure that 
# all features are given equal weight in the distance metric, 
# resulting in better performance of the algorithm.

# Scale the training and testing data
# (without the factor and the response variable).
train_data_scaled = scale(train_data[, -c(1, 53)])
test_data_scaled = scale(test_data[, -c(1, 53)])

################### START: Select the optimal number of NN #####################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Define the grid of K values to try.
k_grid = data.frame(k = 1:20)

# Define the training control with 10-fold cross-validation.
train_control = trainControl(method = "cv", number = 10)

# Fit the KNN model with different values of K.
knn_model = train(train_data_scaled, train_data$trump_over_perc50,
                  method = "knn",
                  trControl = train_control,
                  tuneGrid = k_grid,
                  preProcess = c("center", "scale"))

# Print the results.
print(knn_model)
# Results: 5 NN and then 4.

# Plot the cross-validation results.
plot(knn_model)

# Summary: We are going to perform the KNN with 5 and 4 NN (the two optima 
# number of neighbors) and with both scaled and not scaled datasets.

################### END: Select the optimal number of NN #######################

############################ START: KNN with k = 5 #############################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Fit KNN on the scaled data.
knn5_model = knn(train = train_data_scaled, test = test_data_scaled, 
                 cl = train_data$trump_over_perc50, k = 5)

# Create a confusion matrix.
table(test_data$trump_over_perc50, knn5_model)

# KNN (k = 5) model metrics (Confusion matrix).
knn5_metric = confusionMatrix(knn5_model, 
                              as.factor(test_data$trump_over_perc50),
                              positive = "1")

knn5_metric 
# 69.20

# Prepare the ROC curve (TPR and FPR from predictions).
knn5_pred = prediction(as.integer(knn5_model)-1, test_data$trump_over_perc50)
knn5_roc = performance(knn5_pred, "tpr", "fpr")

# Output the ROC curve.
plot(knn5_roc, col = "steelblue3", main = "ROC Curve - KNN 5", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label. 

abline(0, 1, lty = 2)  # Add diagonal reference line.

############################ END: KNN with k = 5 ###############################

###################### START: KNN with k = 5 (not scaled) ######################

# Now we will provide the not scaled datasets - model results.
# The scaled ones are - as expected - way better.

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Fit KNN on the not scaled data.
knn5_notScaled_model = knn(train = train_data[,-1], test = test_data[,-1],
                           cl = train_data$trump_over_perc50, k = 5)

# Create a confusion matrix.
table(test_data$trump_over_perc50, knn5_notScaled_model)

# KNN (k = 5) model metrics (Confusion matrix) - NOT scaled model.
knn5_notScaled_metric = confusionMatrix(knn5_notScaled_model,
                                        as.factor(test_data$trump_over_perc50),
                                        positive = "1")
knn5_notScaled_metric 
# 56.04

# Prepare the ROC curve (TPR and FPR from predictions).
knn5notScaled_pred = prediction(as.integer(knn5_notScaled_model)-1, 
                                test_data$trump_over_perc50)

knn5notScaled_roc = performance(knn5notScaled_pred, "tpr", "fpr")

# Output the ROC curve.
plot(knn5notScaled_roc, col = "steelblue3", main = "ROC Curve - KNN 5 (Not Scaled)", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label. 

abline(0, 1, lty = 2)  # Add diagonal reference line.

###################### END: KNN with k = 5 (not scaled) ########################

############################ START: KNN with k = 4 #############################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Fit KNN on the scaled data.
knn4_model = knn(train = train_data_scaled, test = test_data_scaled, 
                 cl = train_data$trump_over_perc50, k = 4)

# Create a confusion matrix.
table(test_data$trump_over_perc50, knn4_model)

# KNN (k = 4) model metrics (Confusion matrix).
knn4_metric = confusionMatrix(knn4_model, 
                              as.factor(test_data$trump_over_perc50),
                              positive = "1")

knn4_metric 
# 71.74

# Prepare the ROC curve (TPR and FPR from predictions).
knn4_pred = prediction(as.integer(knn4_model)-1, test_data$trump_over_perc50)
knn4_roc = performance(knn4_pred, "tpr", "fpr")

# Output the ROC curve.
plot(knn4_roc, col = "steelblue3", main = "ROC Curve - KNN 4", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label. 

abline(0, 1, lty = 2)  # Add diagonal reference line.

############################ END: KNN with k = 4 ###############################

###################### START: KNN with k = 4 (not scaled) ######################

# Now we will provide the not scaled datasets - model results.
# the scaled ones are - as expected - way better.

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Fit KNN on the not scaled data.
knn4_notScaled_model = knn(train = train_data[,-1], test = test_data[,-1],
                           cl = train_data$trump_over_perc50, k = 4)

# Create a confusion matrix.
table(test_data$trump_over_perc50, knn4_notScaled_model)

# KNN (k = 4) model metrics (Confusion matrix) - NOT scaled model.
knn4_notScaled_metric = confusionMatrix(knn4_notScaled_model,
                                        as.factor(test_data$trump_over_perc50),
                                        positive = "1")
knn4_notScaled_metric 
# 55.07

# Prepare the ROC curve (TPR and FPR from predictions).
knn4notScaled_pred = prediction(as.integer(knn4_notScaled_model)-1,
                                test_data$trump_over_perc50)
knn4notScaled_roc = performance(knn4notScaled_pred, "tpr", "fpr")

# Output the ROC curve.
plot(knn4notScaled_roc, col = "steelblue3", main = "ROC Curve - KNN 4 (Not Scaled)", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label. 

abline(0, 1, lty = 2)  # Add diagonal reference line.

###################### END: KNN with k = 4 (not scaled) ########################

########################## END: KNN Model ######################################

########################## START: Models Comparison ############################

###################### START: Balance/Imbalanced Check #########################

# Dataframe's balance/imbalance check.
# Check the balance of the us_elections_2016 dataframe.
trumpY = length(which(us_elections_2016$trump_over_perc50 == 1))
trumpN = length(which(us_elections_2016$trump_over_perc50 == 0))
trump_balance = trumpY/(trumpY + trumpN) * 100
trump_balance
# 40.46% = trump_over_perc50 == 1 (Trump wins majority)
# 59.54% = trump_over_perc50 == 0 (Trump lost majority)

# Check the balance of the train dataframe.
trumpY = length(which(train_data$trump_over_perc50 == 1))
trumpN = length(which(train_data$trump_over_perc50 == 0))
trump_balance = trumpY/(trumpY + trumpN) * 100
trump_balance
# 41.34% = trump_over_perc50 == 1 (Trump wins majority)
# 58.66% = trump_over_perc50 == 0 (Trump lost majority)

# Check the balance of the test dataframe.
trumpY = length(which(test_data$trump_over_perc50 == 1))
trumpN = length(which(test_data$trump_over_perc50 == 0))
trump_balance = trumpY/(trumpY + trumpN) * 100
trump_balance
# 39.41% = trump_over_perc50 == 1 (Trump wins majority)
# 60.59% = trump_over_perc50 == 0 (Trump lost majority)

# Summary: 
# As the data is somewhat imbalanced, it is important to consider measures
# beyond accuracy when evaluating model performance.
# Sensitivity and specificity are relevant measures in this case, as they 
# reflect the ability of the model to correctly identify Trump's victories 
# and losses. In the case of imbalanced data, precision is also a relevant
# measure, as it reflects the model's ability to correctly identify Trump's
# victories among all predicted victories.
# Therefore, it is important to consider multiple performance metrics 
# and not rely solely on accuracy in this scenario.

####################### END: Balance/Imbalanced Check ##########################

##################### START: Adjusted Rand Index ###############################

# Create the allmodels vector to make the comparisons.
# allmodels = cbind(dt_class, nb_class, svm_class, knn4_model,
#                   knn4_notScaled_model, knn5_model, knn5_notScaled_model)

# Create the allmodels vector to make the comparisons.
allmodels = cbind(fitted.results, dt_class, nb_class, svm_class, 
                  knn4_model, knn5_model)

# Assign names to the models.
# colnames(allmodels) = c('Decision Tree', 'Naive Bayes', 'SVM', 'KNN 4', 
#                         'KNN 4 Not Scaled', 'KNN 5', 'KNN 5 Not Scaled')

# Assign names to the models.
colnames(allmodels) = c('Logistic Regression', 'Decision Tree', 'Naive Bayes',
                        'SVM', 'KNN 4', 'KNN 5')

# # Compute adjusted Rand Index for all models.
# ari_metric = apply(allmodels, 2, function(x){ 
#   adjustedRandIndex(x, test_data$trump_over_perc50)})

# Compute adjusted Rand Index for all models.
ari_metric = apply(allmodels , 2, function(x){ 
  adjustedRandIndex(x, test_data$trump_over_perc50)})

# Preview the results.
# ari_metric

# Round the values to two decimal places.
ari_metric = round(ari_metric, 2)

# Preview the results.
ari_metric

# Results: 
# Decision Tree    Naive Bayes    SVM 
# 0.56               0.89         0.60 
# 
# KNN 4     KNN 4 Not Scaled      KNN 5 
# 0.86          0.00 (0.004)      0.91 
# 
# KNN 5 Not Scaled    Logistic Regression 
# 0.01 (0.006)        0.58

# Plot only the scaled KNNs since the other two are not relevant and have
# way lower ARI values.

# Output the results as a barplot.
# par(mar = c(4,6,3,1))
# barplot(ari_metric[order(ari_metric, decreasing = T)], 
#         horiz = TRUE, las = 2, xlab = 'Adjusted Rand Index', xlim = c(0, 1))

# Define a vector of colors to use for each model.
model_colors = c('steelblue', 'darkorange', 'forestgreen',
                 'firebrick', 'mediumorchid', 'goldenrod')

# Set the plot margins.
par(mar = c(4, 10, 3, 1))

# Create the barplot.
barplot(
  ari_metric[order(ari_metric, decreasing = TRUE)], # Order the ARI values in decreasing order.
  horiz = TRUE, # Plot the bars horizontally.
  las = 2, # Rotate the y-axis labels 90 degrees.
  xlab = 'Adjusted Rand Index', # Label the x-axis.
  xlim = c(0, 1), # Set the limits of the x-axis.
  col = model_colors, # Set the bar colors based on the model_colors vector.
  border = NA, # Remove the border around the bars.
  main = 'Adjusted Rand Index Barplot', # Add a main title to the plot.
  cex.main = 1.2, # Increase the font size of the main title.
  cex.axis = 0.8, # Decrease the font size of the axis labels.
  font.axis = 2, # Use bold font for the axis labels.
  font.lab = 2 # Use bold font for the x-axis label.
)

# Add a legend to the plot.
legend(
  'topright', # Position the legend in the top-right corner of the plot.
  legend = c('KNN 5', 'Naive Bayes', 'KNN 4', 'SVM',
             'Logistic Regression', 'Decision Tree'), # Labels for each model.
  fill = model_colors, # Set the fill colors based on the model_colors vector.
  border = NA, # Remove the border around the legend.
  cex = 0.8 # Decrease the font size of the legend labels.
)


##################### END: Adjusted Rand Index #################################

########################### START: Measures Table ##############################

# Define the test data (excluding the response variable).
test_data2 = test_data[, -53]

# Define the test labels.
test_labels = as.factor(test_data[, 53])

# Important note !!!
# R highlights as the positive class the 0 instead of the 1.
# We will change that!

# Define the confusion matrix function.
confusionMatrixTable = function(model_name, actuals, predicted) {
  
  # Create the confusion matrix - with positive being the 1 
  # and not the default 0.
  cm = confusionMatrix(actuals, predicted, positive = "1")
  
  # Extract the accuracy score, precision, recall (sensitivity),
  # specificity, and F1 score for the "1" class.
  accuracy = round(cm$overall[1], 3)
  precision = round(cm$byClass['Pos Pred Value'], 3)
  recall = round(cm$byClass['Sensitivity'], 3)
  specificity = round(cm$byClass['Specificity'], 3)
  f1_score = round(2 * precision * recall / (precision + recall), 3)
  
  # Return a named vector with the accuracy, precision, recall,
  # specificity, and F1 score.
  c(Model = model_name,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1_Score = f1_score)
}

# Generate a table of accuracy scores, precision, recall, specificity, and F1 scores.
results = rbind(
  confusionMatrixTable("Logistic Regression", test_labels, as.factor(fitted.results)),
  confusionMatrixTable("Decision Tree", test_labels, dt_class),
  confusionMatrixTable("Naive Bayes", test_labels, nb_class),
  confusionMatrixTable("SVM", test_labels, svm_class),
  confusionMatrixTable("KNN 4", test_labels, knn4_model),
  confusionMatrixTable("KNN 5", test_labels, knn5_model)
)

# Adjust the table.
rownames(results) = results[, 1]
colnames(results) = c("Model", "Accuracy", "Precision", "Recall/Sensitivity",
                      "Specificity", "F1 Score")

# Print the table.
View(results)

# Positive class --> 1 not 0.
# Logistic and SVM (NB has violated assumptions).

########################### END: Measures Table ################################

########################## START: Brier Score ##################################

# Create the Brier Score function.
brier.score = function(y, yhat)
{
  mean((y - yhat)^2)
}

# Calculate the Brier Score for each model 
# (transform the factors to binary numeric).
brier_scores = c(
  brier.score(fitted.results, as.numeric(test_labels)- 1),
  brier.score(as.numeric(dt_class) - 1, as.numeric(test_labels)- 1),
  brier.score(as.numeric(nb_class) - 1, as.numeric(test_labels)- 1),
  brier.score(as.numeric(svm_class) - 1, as.numeric(test_labels)- 1),
  brier.score(as.numeric(knn4_model) - 1, as.numeric(test_labels)- 1),
  brier.score(as.numeric(knn5_model) - 1, as.numeric(test_labels)- 1)
)

# Print the Brier Scores.
brier_scores

# Assign names to the Brier Scores.
model_names = c("Logistic Regression", "Decision Tree",
                "Naive Bayes", "SVM", "KNN 4", "KNN 5")
names(brier_scores) = model_names

# Create a table of the Brier Scores.
brier_table = data.frame("Brier Score" = brier_scores)
colnames(brier_table) = "Brier Score"

# Preview the results.
View(brier_table)

########################### END: Brier Score ###################################

############################ START: Lift Curves ################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Create the lift curves for the models.
# Logistic Regression.
lr_prob = predict(model_aic, newdata = test_data, type = "response")

# Decision Tree.
dt_prob = predict(dt_model, newdata = test_data, type = "prob")

# Naive Bayes.
nb_prob = predict(nb_model, newdata = test_data, type = "raw")

# SVM.
svm_prob = predict(svm_model, newdata = test_data, type = "prob")

# Create the lift_df dataframe with the sorted lift curves.
lift_df = data.frame(Class = as.factor(test_data$trump_over_perc50),
                            lift_log = sort(lr_prob, decreasing = TRUE),
                            lift_dt = sort(dt_prob[, 2], decreasing = TRUE),
                            lift_nb = sort(nb_prob[, 2], decreasing = TRUE),
                            lift_svm = sort(svm_prob, decreasing = TRUE))

# Create the lift curve comparison object. 
lift_curve_com = lift(Class ~ lift_log + lift_dt + lift_nb + lift_svm, 
                      data = lift_df)

lift_data = split(lift_curve_com$data, lift_curve_com$data$liftModelVar)

# Plot lift curves with ggplot.
# ggplot() +
#   geom_line(data = lift_data[[1]], aes(CumTestedPct, CumEventPct, color = "lift_log"), show.legend = TRUE) +
#   geom_line(data = lift_data[[2]], aes(CumTestedPct, CumEventPct, color = "lift_dt"), show.legend = TRUE) +
#   geom_line(data = lift_data[[3]], aes(CumTestedPct, CumEventPct, color = "lift_nb"), show.legend = TRUE) +
#   geom_line(data = lift_data[[4]], aes(CumTestedPct, CumEventPct, color = "lift_svm"), show.legend = TRUE) +
#   geom_line(data = lift_data[[5]], aes(CumTestedPct, CumEventPct, color = "lift_knn4"), show.legend = TRUE) +
#   geom_line(data = lift_data[[6]], aes(CumTestedPct, CumEventPct, color = "lift_knn5"), show.legend = TRUE) +
#   xlab("% Samples tested") +
#   ylab("% Samples found") +
#   scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442"), 
#                      breaks = c("lift_log", "lift_dt", "lift_nb", "lift_svm"), 
#                      labels = c("Logistic Regression", "Decision Tree", "Naive Bayes", "SVM")) +
#   geom_polygon(data = data.frame(x = c(0, lift_curve_com$pct, 100, 0),
#                                  y = c(0, 100, 100, 0)),
#                aes(x = x, y = y), alpha = 0.1)

# Decide to plot only the top 4 models (even keep Naive Bayes).
# Without the KNN models.
ggplot() +
  geom_line(data = lift_data[[1]], 
            aes(CumTestedPct, CumEventPct, color = "Logistic Regression"),
            size = 1.7, show.legend = TRUE) +
  geom_line(data = lift_data[[2]], 
            aes(CumTestedPct, CumEventPct, color = "Decision Tree"), 
            size = 1.5, show.legend = TRUE) +
  geom_line(data = lift_data[[3]], 
            aes(CumTestedPct, CumEventPct, color = "Naive Bayes"), 
            size = 2, show.legend = TRUE) +
  geom_line(data = lift_data[[4]],
            aes(CumTestedPct, CumEventPct, color = "SVM"),
            size = 1.5, show.legend = TRUE) +
  xlab("% Samples tested") +
  ylab("% Samples found") +
  scale_color_manual(name = "Model", 
                     values = c("Logistic Regression" = "#E69F00",
                                "Decision Tree" = "#56B4E9",
                                "Naive Bayes" = "#FF0000", 
                                "SVM" = "#F0E442")) +
  ggtitle("Comparison of Lift Curves for Top-4 Models") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_polygon(data = data.frame(x = c(0, lift_curve_com$pct, 100, 0),
                                 y = c(0, 100, 100, 0)),
               aes(x = x, y = y), alpha = 0.1)

############################ END: Lift Curves ################################

############################ START: ROC Curves #################################

# Define labels for the legend.
labels = c("Logistic Regression", "Decision Tree", "Naive Bayes",
           "SVM", "KNN4", "KNN5")

# Using the created plots to make a collective one.
plot(log_roc, col = "steelblue", main = "ROC Curves - Models Comparisons", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     xlim = c(0, 1), ylim = c(0, 1), 
     lwd = 2,  # Set axis labels, limits, and line width.
     cex.main = 1.2, # Increase the font size of the main title.
     cex.axis = 0.8, # Decrease the font size of the axis labels.
     font.axis = 2, # Use bold font for the axis labels.
     font.lab = 2) # Use bold font for the x-axis label.
plot(dt_roc, col = "darkorange", add = TRUE)
plot(nb_roc, col = "forestgreen", add = TRUE)
plot(svm_rocs, col = "firebrick", add = TRUE)
plot(knn4_roc, col = "mediumorchid", add = TRUE)
plot(knn5_roc, col = "pink", add = TRUE)
abline(0, 1, lty = 2)  # Add diagonal reference line.

# Add a legend to the plot.
legend("bottomright", legend = labels, col = c("steelblue", "darkorange",
                                               "forestgreen", "firebrick",
                                               "mediumorchid", "pink"),
                                               lty = 1, lwd = 2, cex = 0.5)

############################# END: ROC Curves ##################################

############################ START: AUC values #################################

test_labels = as.factor(test_data$trump_over_perc50)

# Logistic Regression.
lr_prob = predict(model_log, newdata = test_data, type = "raw")
lr_pred = prediction(lr_prob, test_data$trump_over_perc50)
lr_auc = performance(lr_pred, "auc")@y.values[[1]]

# Check
# log_auc = roc(test_data$trump_over_perc50, as.numeric(lr_prob))
# Same results!

# Decision Tree.
dt_prob = predict(dt_model, newdata = test_data, type = "prob")
dt_pred = prediction(as.numeric(dt_prob [, 2]) - 1, test_data$trump_over_perc50)
dt_auc = performance(dt_pred, "auc")@y.values[[1]]

# Naive Bayes.
nb_prob = predict(nb_model, newdata = test_data, type = "class")
nb_pred = prediction(as.numeric(nb_prob) - 1 , as.numeric(test_labels) - 1)
nb_auc = performance(nb_pred, "auc")@y.values[[1]]

# SVM.
svm_prob = predict(svm_model, newdata = test_data, type = "raw")
svm_pred = prediction(as.numeric(svm_prob) - 1, as.numeric(test_labels) - 1)
svm_auc = performance(svm_pred, "auc")@y.values[[1]]

# KNN 4.
knn4_pred = prediction(as.numeric(knn4_model) - 1, as.numeric(test_labels) - 1)
knn4_auc = performance(knn4_pred, "auc")@y.values[[1]]

# KNN 5.
knn5_pred = prediction(as.numeric(knn5_model) - 1, as.numeric(test_labels) - 1)
knn5_auc = performance(knn5_pred, "auc")@y.values[[1]]

# Create a data frame with the model names and AUCs.
models_df = data.frame(Model = c("Logistic Regression", "Decision Tree", 
                                 "Naive Bayes", "SVM", "KNN4", "KNN5"),
                       AUC = c(lr_auc, dt_auc, nb_auc, svm_auc,
                               knn4_auc, knn5_auc))

# Print the table.
View(models_df)

# Create a data frame with the model names and AUCs.
models_df_top3 = data.frame(Model = c("Logistic Regression", "Decision Tree", 
                                 "SVM"),
                       AUC = c(lr_auc, dt_auc, svm_auc))

# Print the table.
View(models_df_top3)

############################# END: AUC values ##################################

########################## END: Models Comparison ##############################

################################################################################
################################################################################
######################### START: Clustering Part II ############################
################################################################################
################################################################################
########################## START: Dataset Preparation ##########################

# Load Excel file.
df_clustering = read_excel("stat BA II project I.xlsx", sheet = "county_facts")

# Remove rows with NAs (states/US cumulative).
df_clustering = na.omit(df_clustering)

# Remove not necessary columns (fips, state, and county).
df_clustering = df_clustering[, -c(1, 2, 3)]

# Validate the changes.
# View(df_clustering)

# Output the data frame to a CSV-file, for testing purposes.
# write.csv(df_clustering, file = "us_elections_2016_clustering.csv",
# row.names = FALSE)

######################### END: Dataset Preparation #############################

################### START: Demographics Data Preparation #######################

# File Checkpoint - Load in the data.
df_clustering = read.csv("us_elections_2016_clustering.csv", header = TRUE)

# Creates a list of character strings containing the 
# demographics related variable names.
demographics_names = list("PST045214", "PST040210", "PST120214", "POP010210",
                          "AGE135214", "AGE295214", "AGE775214", "SEX255214",
                          "RHI125214", "RHI225214", "RHI325214", "RHI425214",
                          "RHI525214", "RHI625214", "RHI725214", "RHI825214",
                          "POP715213", "POP645213", "POP815213", "EDU635213",
                          "EDU685213", "VET605213")


# Creates a list of character strings containing the 
# demographics related variable names.
demographics_matrix = names(df_clustering) %in% demographics_names

# Create the demographics dataframe.
demographics = df_clustering[demographics_matrix]

# Create the economics dataframe.
economics = df_clustering[!demographics_matrix]

# Remove the not necessary variables.
rm(demographics_names, demographics_matrix) 

################ START: Correlations Calculations & Scaling ####################

# Define a function to calculate and visualize correlations (over 0.7).
calculate_correlations = function(data = df, sig_level = 0.7)
{
  
  # Convert character columns to factors and all factors to numeric values.
  df_cor = data %>% 
    mutate_if(is.character, as.factor) %>% 
    mutate_if(is.factor, as.numeric)
  
  # Calculate correlations and round to three decimal places.
  corr = round(cor(df_cor), 3)
  
  # Set lower triangle and diagonal values to NA to remove duplicates.
  corr[lower.tri(corr, diag = TRUE)] = NA
  
  # Remove perfect correlations.
  corr[corr == 1] = NA
  
  # Convert correlations to a 3-column data frame.
  corr_df = as.data.frame(as.table(corr))
  
  # Remove rows with NA values.
  corr_df = na.omit(corr_df)
  
  # Select correlations with absolute value greater 
  # than the specified significance level.
  corr_df = subset(corr_df, abs(Freq) > sig_level)
  
  # Sort correlations by absolute value in descending order.
  corr_df = corr_df[order(-abs(corr_df$Freq)),]
  
  # Print the correlation table.
  print(corr_df)
  
  # Convert correlation data frame to a matrix for visualization.
  corr_matrix = reshape2::acast(corr_df, Var1 ~ Var2, value.var = "Freq")
  
  # Create a correlation plot.
  corrplot(corr_matrix, method = "square", tl.col = "black", 
           tl.srt = 45, is.corr = FALSE, tl.cex = 0.8, 
           cl.cex = 0.8 , na.label = " ",  addCoef.col = "black")

}

# Apply the above created function.
calculate_correlations(demographics)

# Var1      Var2   Freq
# 463 PST045214 VET605213  0.927
# 464 PST040210 VET605213  0.925
# 466 POP010210 VET605213  0.925
# 411 RHI725214 POP815213  0.899
# 115 AGE135214 AGE295214  0.862
# 207 RHI125214 RHI225214 -0.828
# 414 POP645213 POP815213  0.812
# 339 RHI125214 RHI825214  0.769

# Validate that no correlations are existing.
# calculate_correlations(demographics[,-c(5,9,19,22)])
# No correlation exists!

# Parse the results to a new dataframe.
demographics_noncor = demographics[,-c(5,9,19,22)]

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Scale the data.
demographics_scaled = scale(demographics_noncor)
# demographics_scaled_matrix = as.matrix(demographics_scaled)

################# END: Correlations Calculations & Scaling #####################
#################### END: Demographics Data Preparation ########################

##################### START: Hierarchical Clustering ###########################

# Complete Method with Euclidean Distance.
# Perform hierarchical clustering using the complete method with Euclidean 
# distance as the similarity metric.
hcl_complete = hclust(dist(demographics_scaled, method = 'euclidean'),
                      method = "complete")

# Plot the dendogram of the complete method model.
plot(hcl_complete)

# Indicate the three clusters in the dendogram of the complete method model 
# using a red border.
rect.hclust(hcl_complete, k = 3, border="red")

# Calculate the average silhouette coefficient
# for different numbers of clusters.
sil_width = c(NA)
for (i in 2:10) {
  clust = cutree(hcl_complete, i)
  sil = silhouette(clust, dist(demographics_scaled, method = "euclidean"))
  sil_width[i] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers
# of clusters to determine the optimal number of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, we see that 5 clusters are the most appropriate option.
# Plot the silhouette of the complete method model 
# with 5 clusters to visualize the cluster quality.
plot(silhouette(cutree(hcl_complete, k = 5), 
                dist(demographics_scaled), method = "euclidean"),
     main ='Complete K = 5', border = "NA")

# Results: 0.75
# The average silhouette width of 0.75 suggests high cluster quality,
# but it is one as this method suggest, thus we do not trust it.

# Centroid Method with Euclidean Distance.
# Perform hierarchical clustering using the centroid method 
# with Euclidean distance as the similarity metric.
hcl_centroid = hclust(dist(demographics_scaled, method = 'euclidean'), 
                      method = "centroid")

# Plot the dendogram of the centroid method model.
plot(hcl_centroid)

# Indicate the three clusters in the dendogram 
# of the centroid method model using a red border.
rect.hclust(hcl_centroid, k = 3, border = "red")

# Calculate the average silhouette coefficient 
# for different numbers of clusters.
sil_width = c(NA)
for (i in 2:10) {
  clust = cutree(hcl_centroid, i)
  sil = silhouette(clust, dist(demographics_scaled, method = "euclidean"))
  sil_width[i] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers 
# of clusters to determine the optimal number of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, we see that 6 clusters are the most appropriate option.
# Plot the silhouette of the centroid method model 
# with 6 clusters to visualize the cluster quality.
plot(silhouette(cutree(hcl_centroid, k = 6), 
                dist(demographics_scaled),
                method = "euclidean"), main = 'Centroid K = 6', border = "NA")

# Results: 0.78
# The average silhouette width of 0.78 suggests high cluster quality,
# but it is one as this method suggest, thus we do not trust it.

# Ward Method with Euclidean Distance.
# Perform hierarchical clustering using the Ward method 
# with Euclidean distance as the similarity metric.
dist_matrix = dist(demographics_scaled, method = "euclidean")
hcl_ward = hclust(dist_matrix, method = "ward.D2")

# Plot the dendogram of the Ward method model.
plot(hcl_ward)

# Indicate the three clusters in the dendogram of
# the Ward method model using a red border.
rect.hclust(hcl_ward, k = 8, border = "red")

# Calculate the silhouette coefficient for different numbers of clusters.
sil_width = c(NA)
for (num_clusters in 2:10) {
  cluster_assignment = cutree(hcl_ward, num_clusters)
  sil = silhouette(cluster_assignment, dist(demographics_scaled, method = "euclidean"))
  sil_width[num_clusters] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, 8 clusters seem to be the most appropriate option.
plot(silhouette(cutree(hcl_ward, k = 8), 
                dist(demographics_scaled), method = "euclidean"),
     main = 'Ward K = 8', border = "NA")

# Results: 0.19
# Average silhouette value of 0.19 with more homogeneous clusters.

# Ward Method with Manhattan Distance.
# Perform hierarchical clustering using the Ward method 
# with Euclidean distance as the similarity metric.
dist_matrix = dist(demographics_scaled, method = "man")
hcl_ward_man = hclust(dist_matrix, method = "ward.D2")

# Plot the dendogram of the Ward method model.
plot(hcl_ward_man)

# Indicate the three clusters in the dendogram of
# the Ward method model using a red border.
rect.hclust(hcl_ward_man, k = 6, border = "red")

# Calculate the silhouette coefficient for different numbers of clusters.
sil_width = c(NA)
for (num_clusters in 2:10) {
  cluster_assignment = cutree(hcl_ward_man, num_clusters)
  sil = silhouette(cluster_assignment, dist(demographics_scaled, method = "man"))
  sil_width[num_clusters] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, 6 clusters seem to be the most appropriate option.
plot(silhouette(cutree(hcl_ward, k = 6), 
                dist(demographics_scaled), method = "man"),
     main = 'Ward Man K = 6', border = "NA")

# Results: 0.21
# Average silhouette value of 0.21 with more homogeneous clusters.

# Single Method.
# Clustering with single method and euclidean distance.
distance_matrix = dist(demographics_scaled, method = "euclidean")
single_hclustering = hclust(distance_matrix, method = "single")
plot(single_hclustering)
rect.hclust(single_hclustering, k = 5, border="red")

# Calculate the silhouette coefficient for different numbers of clusters.
sil_width = c(NA)
for (num_clusters in 2:10) {
  cluster_assignment = cutree(single_hclustering, num_clusters)
  sil = silhouette(cluster_assignment, dist(demographics_scaled, method = "euclidean"))
  sil_width[num_clusters] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, 5 clusters seem to be the most appropriate option.
plot(silhouette(cutree(single_hclustering, k = 5), dist(demographics_scaled),
                method = "euclidean"),main = 'Single K = 5', border = "NA")


# Results: 0.78
# The average silhouette width of 0.78 suggests high cluster quality,
# but it is one as this method suggest, thus we do not trust it.

# Summary: 
# Based on the analysis of the clustering results using the ward, complete,
# centroids, and single methods, it was observed that the ward method presented
# a lower value of the average silhouette score compared to the other method.
# However, the ward method was found to produce more homogeneous clusters.
# It is important to note that the main objective of this clustering analysis
# was not solely focused on pattern recognition in the dataset,
# but rather on identifying distinct groups. Therefore, despite the lower
# average silhouette score, the ward method was deemed the most
# appropriate method to achieve the desired outcome.

###################### END: Hierarchical Clustering ############################

####################### START: K-Means Clustering ##############################

########################## START: Elbow Plot ###################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Compute and plot within-cluster sum of squares (WSS) for k = 2 to k = 15.
# Set the maximum number of clusters to 15.
kmax = 15 

# Compute the WSS for each value of k using k-means clustering.
wss = sapply(1:kmax, function(k) {
  # Perform k-means clustering on the scaled demographics data
  # using k clusters, 50 random starts, and a maximum of 15 iterations
  kmeans(demographics_scaled, k, nstart = 20, iter.max = 15)$tot.withinss
  }
)

# Print the WSS values for each value of k.
wss

# Plot the WSS values against the number of clusters.
plot(1:kmax, wss, type = "b", pch = 19,
     main = "Elbow Plot for K-Means Clustering",
     xlab = "Number of Clusters", 
     ylab = "Within-cluster Sum of Squares (WSS)")

# Results: K-Means with K = 6.

########################### END: Elbow Plot ####################################

######################### START: K-Means Models ################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=6.
kmeans_6 = kmeans(demographics_scaled, centers = 6, nstart = 25)

# Print K-Means results.
# print(kmeans_6)

# Examine K-Means object structure.
# str(kmeans_6)

# Aggregate the mean of each variable by cluster.
dd_mean_6 = aggregate(demographics_scaled, 
                      by = list(Cluster = kmeans_6$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_6 = cbind(demographics_scaled, Cluster = kmeans_6$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=4.
kmeans_4 = kmeans(demographics_scaled, centers = 4, nstart = 25)

# Print K-Means results.
# print(kmeans_4)

# Examine K-Means object structure.
# str(kmeans_4)

# Aggregate the mean of each variable by cluster.
dd_mean_4 = aggregate(demographics_scaled,
                      by = list(Cluster = kmeans_4$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_4 = cbind(demographics_scaled, Cluster = kmeans_4$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=5.
kmeans_5 = kmeans(demographics_scaled, centers = 5, nstart = 25)

# Print K-Means results.
# print(kmeans_5)

# Examine K-Means object structure.
# str(kmeans_5)

# Aggregate the mean of each variable by cluster.
dd_mean_5 = aggregate(demographics_scaled, 
                      by = list(Cluster = kmeans_5$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_5 = cbind(demographics_scaled, Cluster = kmeans_5$cluster)

########################## END: K-Means Models #################################

######################### START: K-Means Plots #################################

# Visualize the data using pairs plot with colors assigned by cluster.
# Visualize K-Means with 4 clusters.
pairs(demographics_scaled, col = kmeans_4$cluster) 

# Visualize K-Means with 5 clusters.
pairs(demographics_scaled, col = kmeans_5$cluster) 

# Visualize K-Means with 6 clusters.
pairs(demographics_scaled, col = kmeans_6$cluster) 

# Output format to fit all 3 diagrams.
par(mfrow=c(1,3))

# Plot the silhouette diagrams.
plot(silhouette(kmeans_4$cluster, dist(demographics_scaled)), col = 2:5, 
     main ='K-Means4 - Ward', border = 'NA') 
# 0.26 (2 negatives).

plot(silhouette(kmeans_5$cluster, dist(demographics_scaled)), 
     col = 2:6, main ='K-Means5 - Ward', border = 'NA')
# 0.25 (1 negative).

plot(silhouette(kmeans_6$cluster, dist(demographics_scaled)),
     col = 2:7, main ='K-Means6 - Ward', border = 'NA') 
# 0.26 (1 negative).

# Results:
# K = 4 and K = 6 have the highest avg. silhouette width, however the 
# K = 4 model presents two negative widths in comparison with the other two 
# models that present only one case. Therefore K-Means with 6 seems to perform
# better than the other models.

# Calculations of WSS for the three models.
kmm4_wss = sum(kmeans_4$withinss)
kmm5_wss = sum(kmeans_5$withinss)
kmm6_wss = sum(kmeans_6$withinss)

# Print the findings.
print(kmm4_wss)
# 38897.1

print(kmm5_wss)
# 35553.66

print(kmm6_wss) 
# 32654.19

# Results: K-Means with K = 6 is better since it has the lower WSS of 32654.19.

########################## END: K-Means Plots ##################################

##################### START: Adjusted Rand Index ###############################

# Adjusted Rand Index for the K-Means models.
adjustedRandIndex(kmeans_4$cluster, kmeans_5$cluster)
# 0.8515281

adjustedRandIndex(kmeans_4$cluster, kmeans_6$cluster)
# 0.8482961

adjustedRandIndex(kmeans_5$cluster, kmeans_6$cluster)
# 0.9831185

# Define a vector of colors to use for each model.
model_colors = c('steelblue', 'darkorange', 'forestgreen')

# Set the plot margins.
par(mar = c(3, 9, 4, 5))

# Create the barplot.
barplot(
  c(0.8515281, 0.8482961, 0.9831185), # ARI values for the K-Means models.
  horiz = T, # Plot the bars horizontally.
  las = 2, # Rotate the y-axis labels 90 degrees.
  xlab = 'Adjusted Rand Index', # Label the x-axis.
  xlim = c(0, 1), # Set the limits of the x-axis.
  col = model_colors[1:3], # Set the bar colors based on the model_colors vector.
  border = NA, # Remove the border around the bars.
  main = 'Adjusted Rand Index Barplot', # Add a main title to the plot.
  cex.main = 1.2, # Increase the font size of the main title.
  cex.axis = 0.8, # Decrease the font size of the axis labels.
  font.axis = 2, # Use bold font for the axis labels.
  font.lab = 2, # Use bold font for the x-axis label.
  names.arg = c('KMM 4 vs KMM 5', 'KMM 4 vs KMM 6', 'KMM 5 vs KMM 6') # Labels for each model.
)

# Add the percentage of each bar at the end of it.
values = c(0.8515281, 0.8482961, 0.9831185)
percentages = round(values* 100, 1)
text(x = values - 0.05, y = 1:3, labels = paste0(percentages, '%'))

# Add a legend to the plot.
legend(
  'bottomright', # Position the legend in the top-right corner of the plot.
  legend = c('KMM 4 vs KMM 5', 'KMM 4 vs KMM 6', 'KMM 5 vs KMM 6'), # Labels for each model.
  fill = model_colors[1:3], # Set the fill colors based on the model_colors vector.
  border = NA, # Remove the border around the legend.
  cex = 0.8 # Decrease the font size of the legend labels.
)

# Table comparisons.
table(kmeans_4$cluster, kmeans_5$cluster)
table(kmeans_4$cluster, kmeans_6$cluster)
table(kmeans_5$cluster, kmeans_6$cluster)

###################### END: Adjusted Rand Index ################################

########################## START: KNN Plots ####################################

# Output format - plotting area.
par(mfrow = c(1,3))

# Plot kmeans_4 with different colors.
clusplot(demographics_scaled, kmeans_4$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Cluster Analysis with 4 Clusters',
         col.p=kmeans_4$cluster)

# Plot kmeans_5 with different colors.
clusplot(demographics_scaled, kmeans_5$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Cluster Analysis with 5 Clusters',
         col.p=kmeans_5$cluster)

# Plot kmeans_6 with different colors.
clusplot(demographics_scaled, kmeans_6$cluster, 
         color = TRUE, 
         shade=F,
         labels=0,
         lines=0, 
         main='k-Means Cluster Analysis with 6 Clusters',
         col.p=kmeans_6$cluster)

# Results:
# From the plots again we can see that K-Means with 6 clusters is the best one.

########################### END: KNN Plots #####################################

######################### END: K-Means Clustering ##############################

######################### START Models Comparisons #############################

# Set the plotting area.
par(mfrow = c(1,2))
 
# Plot the silhouette plot for k-means clustering with 6 clusters.
plot(silhouette(kmeans_6$cluster,
                dist(demographics_scaled)),
     col = 2:7,
     main ='K-Means (K = 6)', border = 'NA')

# Plot the silhouette plot for hierarchical clustering 
# with 6 clusters using the "Ward - Manhattan" method.
plot(silhouette(cutree(hcl_ward_man, k = 6), dist(demographics_scaled), 
                method = "man"),
     main ='HC Ward Manhattan (K = 6)',
     col = 2:7, border = "NA")

# Create a table showing the count of observations 
# in each cluster for both models.
table(cutree(hcl_ward_man, k = 6), kmeans_6$cluster)

# Output: 
#      1    2    3    4    5    6
# 1    0   65 1569   27    0    1
# 2    0  369   68   31    0    4
# 3    0    0   38  310    0    0
# 4    2   27  135   31    0  214
# 5   44   93    0    1    0    8
# 6    0    9   53    0   44    0

# Set the plotting area.
par(mfrow = c(1,2))

# Plot the K-Means with 6 clusters.
clusplot(demographics_scaled, kmeans_6$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Clustering (K = 6)',
         col.p = kmeans_6$cluster)


# Plot the Hierarchical Clustering with the Ward Manhattan method and 6 clusters.
clusplot(demographics_scaled, cutree(hcl_ward_man, k = 6), 
         color = T, 
         shade = F,
         labels = 0,
         lines = 0, 
         main = 'Hierarchical Clustering - Ward Manhattan Method (K = 6)',
         col.p = cutree(hcl_ward_man, k = 6))

# Results: 
# he K-Means method with 6 clusters provides better clustering results 
# than the Hierarchical method using the Ward Manhattan distancMetrics.

########################## END Models Comparisons ##############################

########################## START: ANOVA ########################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Scale the data.
demographics_scaled2 = scale(demographics)

# Add the cluster from the best clustering model (K-Means with K = 6).
demographics_scaled2 = as.data.frame(cbind(demographics_scaled2,
                                           cluster = kmeans_6$cluster))



# Perform ANOVA to each variable to validate if it
# significantly differs across clusters, utilizing lapply.
anova_result = lapply(demographics_scaled2[,1:ncol(demographics_scaled2)-1], 
                function(x) aov(x ~ kmeans_6$cluster, 
                                data = demographics_scaled2))


# Extract the p-values from the ANOVA results and create a vector.
anova_pvalues = sapply(anova_result, function(x) summary(x)[[1]][["Pr(>F)"]][1])


# Identify the variables with p-values greater than 0.05,
# indicating that they do not significantly contribute to the clustering.

nonsign_vars = names(anova_pvalues[anova_pvalues > 0.05])

nonsign_vars
# [1] "AGE775214"

# Removing the variables that do not significantly contribute to the clustering.
demographics_scaled2 = demographics_scaled2[,-c(7)]

# Repeat the process to further validate.
# No further removal needed!
# Thus we create the clustering models without the variable that ANOVA 
# indicates us to remove.

############################ END: ANOVA ########################################

############################# Cluster Models Again #############################

##################### START: Hierarchical Clustering ###########################

# Ward Method with Euclidean Distance.
# Perform hierarchical clustering using the Ward method 
# with Euclidean distance as the similarity metric.
dist_matrix = dist(demographics_scaled2, method = "euclidean")
hcl_ward = hclust(dist_matrix, method = "ward.D2")

# Plot the dendogram of the Ward method model.
plot(hcl_ward)

# Indicate the three clusters in the dendogram of
# the Ward method model using a red border.
rect.hclust(hcl_ward, k = 9, border = "red")

# Calculate the silhouette coefficient for different numbers of clusters.
sil_width = c(NA)
for (num_clusters in 2:10) {
  cluster_assignment = cutree(hcl_ward, num_clusters)
  sil = silhouette(cluster_assignment, dist(demographics_scaled2, method = "euclidean"))
  sil_width[num_clusters] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, 9 clusters seem to be the most appropriate option.
plot(silhouette(cutree(hcl_ward, k = 9), 
                dist(demographics_scaled2), method = "euclidean"),
     main = 'Ward K = 9', border = "NA")

# Results: 0.3
# Average silhouette value of 0.3 with homogeneous clusters.

# Ward Method with Manhattan Distance.
# Perform hierarchical clustering using the Ward method 
# with Euclidean distance as the similarity metric.
dist_matrix = dist(demographics_scaled2, method = "man")
hcl_ward_man = hclust(dist_matrix, method = "ward.D2")

# Plot the dendogram of the Ward method model.
plot(hcl_ward_man)

# Indicate the three clusters in the dendogram of
# the Ward method model using a red border.
rect.hclust(hcl_ward_man, k = 4, border = "red")

# Calculate the silhouette coefficient for different numbers of clusters.
sil_width = c(NA)
for (num_clusters in 2:10) {
  cluster_assignment = cutree(hcl_ward_man, num_clusters)
  sil = silhouette(cluster_assignment, dist(demographics_scaled2, method = "man"))
  sil_width[num_clusters] = mean(sil[, 3])
}

# Plot the silhouette coefficient for different numbers of clusters.
plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of clusters",
     ylab = "Silhouette Width")

# Based on the plot, 4 clusters seem to be the most appropriate option.
plot(silhouette(cutree(hcl_ward_man, k = 4), 
                dist(demographics_scaled2), method = "man"),
     main = 'Ward Man K = 4', border = "NA")

# Results: 0.34 
# Thus, we select it as the best HClustering model.

###################### END: Hierarchical Clustering ############################

####################### START: K-Means Clustering ##############################

########################## START: Elbow Plot ###################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Compute and plot within-cluster sum of squares (WSS) for k = 2 to k = 15.
# Set the maximum number of clusters to 15.
kmax = 10

# Compute the WSS for each value of k using k-means clustering.
wss = sapply(1:kmax, function(k) {
  # Perform k-means clustering on the scaled demographics data
  # using k clusters, 50 random starts, and a maximum of 15 iterations
  kmeans(demographics_scaled2, k, nstart = 20, iter.max = 15)$tot.withinss
}
)

# Print the WSS values for each value of k.
wss

# Plot the WSS values against the number of clusters.
plot(1:kmax, wss, type = "b", pch = 19,
     main = "Elbow Plot for K-Means Clustering",
     xlab = "Number of Clusters", 
     ylab = "Within-cluster Sum of Squares (WSS)")

# Results: K-Means with K = 8.

########################### END: Elbow Plot ####################################

######################### START: K-Means Models ################################

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=6.
kmeans_6 = kmeans(demographics_scaled2, centers = 6, nstart = 25)

# Print K-Means results.
# print(kmeans_6)

# Examine K-Means object structure.
# str(kmeans_6)

# Aggregate the mean of each variable by cluster.
dd_mean_6 = aggregate(demographics_scaled2, 
                      by = list(Cluster = kmeans_6$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_6 = cbind(demographics_scaled2, Cluster = kmeans_6$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=4.
kmeans_4 = kmeans(demographics_scaled2, centers = 4, nstart = 25)

# Print K-Means results.
# print(kmeans_4)

# Examine K-Means object structure.
# str(kmeans_4)

# Aggregate the mean of each variable by cluster.
dd_mean_4 = aggregate(demographics_scaled2,
                      by = list(Cluster = kmeans_4$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_4 = cbind(demographics_scaled2, Cluster = kmeans_4$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=5.
kmeans_5 = kmeans(demographics_scaled2, centers = 5, nstart = 25)

# Print K-Means results.
# print(kmeans_5)

# Examine K-Means object structure.
# str(kmeans_5)

# Aggregate the mean of each variable by cluster.
dd_mean_5 = aggregate(demographics_scaled2, 
                      by = list(Cluster = kmeans_5$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_5 = cbind(demographics_scaled2, Cluster = kmeans_5$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=8.
kmeans_8 = kmeans(demographics_scaled2, centers = 8, nstart = 25)

# Print K-Means results.
# print(kmeans_8)

# Examine K-Means object structure.
# str(kmeans_8)

# Aggregate the mean of each variable by cluster.
dd_mean_8 = aggregate(demographics_scaled2, 
                      by = list(Cluster = kmeans_8$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_8 = cbind(demographics_scaled2, Cluster = kmeans_8$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=7.
kmeans_7 = kmeans(demographics_scaled2, centers = 7, nstart = 25)

# Print K-Means results.
# print(kmeans_7)

# Examine K-Means object structure.
# str(kmeans_7)

# Aggregate the mean of each variable by cluster.
dd_mean_7 = aggregate(demographics_scaled2, 
                      by = list(Cluster = kmeans_7$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_7 = cbind(demographics_scaled2, Cluster = kmeans_7$cluster)

# Sets a random seed to ensure reproducibility of the analysis.
set.seed(2822212)

# Run K-Means algorithm with K=9.
kmeans_9 = kmeans(demographics_scaled2, centers = 9, nstart = 25)

# Print K-Means results.
# print(kmeans_9)

# Examine K-Means object structure.
# str(kmeans_9)

# Aggregate the mean of each variable by cluster.
dd_mean_9 = aggregate(demographics_scaled2, 
                      by = list(Cluster = kmeans_9$cluster), mean)

# Bind the cluster assignments to the original data frame.
dd_9 = cbind(demographics_scaled2, Cluster = kmeans_9$cluster)

########################## END: K-Means Models #################################

######################### START: K-Means Plots #################################

# Visualize the data using pairs plot with colors assigned by cluster.
# # Visualize K-Means with 4 clusters.
# pairs(demographics_scaled2, col = kmeans_4$cluster) 
# 
# # Visualize K-Means with 5 clusters.
# pairs(demographics_scaled2, col = kmeans_5$cluster) 
# 
# # Visualize K-Means with 6 clusters.
# pairs(demographics_scaled2, col = kmeans_6$cluster) 

# Output format to fit all 3 diagrams.
par(mfrow=c(1,3))

# Plot the silhouette diagrams.
plot(silhouette(kmeans_4$cluster, dist(demographics_scaled2)), col = 2:5, 
     main ='K-Means4', border = 'NA') 
# 0.31 (0 negatives).

plot(silhouette(kmeans_5$cluster, dist(demographics_scaled2)), 
     col = 2:6, main ='K-Means5', border = 'NA')
# 0.3 (1 negative).

plot(silhouette(kmeans_6$cluster, dist(demographics_scaled2)),
     col = 2:7, main ='K-Means6', border = 'NA') 
# 0.31 (1 negative).

plot(silhouette(kmeans_8$cluster, dist(demographics_scaled2)),
     col = 2:9, main ='K-Means8', border = 'NA') 
# 0.3 (0 negative).

plot(silhouette(kmeans_7$cluster, dist(demographics_scaled2)),
     col = 2:8, main ='K-Means7', border = 'NA') 
# 0.30 (0 negative).

plot(silhouette(kmeans_9$cluster, dist(demographics_scaled2)),
     col = 2:10, main ='K-Means9', border = 'NA') 
# 0.16 (0 negative).

# Results:
# K = 4 and K = 6 have the highest avg. silhouette width, however the 
# K = 4 model presents zero negative widths in comparison with the other two 
# models that present only one case. Therefore K-Means with 4 seems to perform
# better than the other models.
# K = 8 ans K = 7 were another two good options but we prefer the model 
# with the 4 clusters since it offers higher avg silhouette value, 
# no negative values and it is simpler.

########################## END: K-Means Plots ##################################

##################### START: Adjusted Rand Index ###############################

# Adjusted Rand Index for the K-Means models.
adjustedRandIndex(kmeans_4$cluster, kmeans_7$cluster)
# 0.6245685

adjustedRandIndex(kmeans_4$cluster, kmeans_8$cluster)
# 0.5855888

adjustedRandIndex(kmeans_7$cluster, kmeans_8$cluster)
# 0.950561

# Define a vector of colors to use for each model.
model_colors = c('steelblue', 'darkorange', 'forestgreen')

# Set the plot margins.
par(mar = c(3, 9, 4, 5))

# Create the barplot.
barplot(
  c(0.8515281, 0.8482961, 0.9831185), # ARI values for the K-Means models.
  horiz = T, # Plot the bars horizontally.
  las = 2, # Rotate the y-axis labels 90 degrees.
  xlab = 'Adjusted Rand Index', # Label the x-axis.
  xlim = c(0, 1), # Set the limits of the x-axis.
  col = model_colors[1:3], # Set the bar colors based on the model_colors vector.
  border = NA, # Remove the border around the bars.
  main = 'Adjusted Rand Index Barplot', # Add a main title to the plot.
  cex.main = 1.2, # Increase the font size of the main title.
  cex.axis = 0.8, # Decrease the font size of the axis labels.
  font.axis = 2, # Use bold font for the axis labels.
  font.lab = 2, # Use bold font for the x-axis label.
  names.arg = c('KMM 4 vs KMM 7', 'KMM 4 vs KMM 8', 'KMM 7 vs KMM 8') # Labels for each model.
)

# Add the percentage of each bar at the end of it.
values = c(0.6245685, 0.5855888, 0.950561)
percentages = round(values* 100, 1)
text(x = values - 0.05, y = 1:3, labels = paste0(percentages, '%'))

# Add a legend to the plot.
legend(
  'bottomright', # Position the legend in the top-right corner of the plot.
  legend = c('KMM 4 vs KMM 7', 'KMM 4 vs KMM 8', 'KMM 7 vs KMM 8'), # Labels for each model.
  fill = model_colors[1:3], # Set the fill colors based on the model_colors vector.
  border = NA, # Remove the border around the legend.
  cex = 0.8 # Decrease the font size of the legend labels.
)

###################### END: Adjusted Rand Index ################################

########################## START: KNN Plots ####################################

# Output format - plotting area.
par(mfrow = c(1,3))

# Plot kmeans_4 with different colors.
clusplot(demographics_scaled2, kmeans_4$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Cluster Analysis with 4 Clusters',
         col.p=kmeans_4$cluster)

# Plot kmeans_7 with different colors.
clusplot(demographics_scaled2, kmeans_7$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Cluster Analysis with 7 Clusters',
         col.p=kmeans_7$cluster)

# Plot kmeans_8 with different colors.
clusplot(demographics_scaled2, kmeans_8$cluster, 
         color = TRUE, 
         shade=F,
         labels=0,
         lines=0, 
         main='k-Means Cluster Analysis with 8 Clusters',
         col.p=kmeans_8$cluster)

# Results:
# From the plots we can see that K-Means with 8 clusters is the best one.
# Thus since the 7 and 8 models identify the outliers and dedicate a cluster to
# them and offer similar values with the 4 model. 
# So the KMeans with K=4 is the best model.

########################### END: KNN Plots #####################################

######################### END: K-Means Clustering ##############################

######################### START Models Comparisons #############################

# Set the plotting area.
par(mfrow = c(1,2))

# Plot the silhouette plot for k-means clustering with 7 clusters.
plot(silhouette(kmeans_7$cluster,
                dist(demographics_scaled2)),
     col = 2:8,
     main ='K-Means (K = 7)', border = 'NA')

# Plot the silhouette plot for hierarchical clustering 
# with 4 clusters using the "Ward - Manhattan" method.
plot(silhouette(cutree(hcl_ward_man, k = 4), dist(demographics_scaled2), 
                method = "man"),
     main ='HC Ward Manhattan (K = 4)',
     col = 2:5, border = "NA")

# Create a table showing the count of observations 
# in each cluster for both models.
table(cutree(hcl_ward_man, k = 4), kmeans_7$cluster)

# Output: 
# 1    2    3    4    5    6    7
# 1  526    1 1803    0    0   17    2
# 2    0    2   21   44    0  373   20
# 3   25   99    0    0    5    0    0
# 4    0    1    0    0    0    0  204

# Set the plotting area.
par(mfrow = c(1,2))

# Plot the K-Means with 6 clusters.
clusplot(demographics_scaled2, kmeans_7$cluster, 
         color = TRUE, 
         shade = F,
         labels = 0,
         lines = 0, 
         main ='K-Means Clustering (K = 7)',
         col.p = kmeans_7$cluster)


# Plot the Hierarchical Clustering with the Ward Manhattan method and 6 clusters.
clusplot(demographics_scaled2, cutree(hcl_ward_man, k = 4), 
         color = T, 
         shade = F,
         labels = 0,
         lines = 0, 
         main = 'Hierarchical Clustering - Ward Manhattan Method (K = 4)',
         col.p = cutree(hcl_ward_man, k = 4))

# Results: 
# he K-Means method with 7 clusters provides better clustering results 
# than the Hierarchical method using the Ward Manhattan.

############################# START: Profiling #################################

################################################################################
# No proper way!!!!!!!!!!!
# # Find the correlations between the economic
# # variables with the above created function.
# # calculate_correlations(economics)
# 
# # Add to the economics data set the cluster columns each observation belongs to.
# df_kmeans7 = as.data.frame(dd_7)
# economics = cbind(economics, df_kmeans7$Cluster)
# 
# # Rename last column.
# colnames(economics)[30] = "Cluster"


################################################################################
# Now we have add the cluster column to the economics dataframe to profile and 
# describe each cluster based on the economics variables.
# 
# # Define a function to calculate and visualize 
# # correlations (over 0.7) with one variable.
# calculate_correlations_2 = function(data = df, variable, sig_level = 0.7)
# {
#   # Convert character columns to factors and all factors to numeric values.
#   df_cor = data %>% 
#     mutate_if(is.character, as.factor) %>% 
#     mutate_if(is.factor, as.numeric)
#   
#   # Select the specified variable and all numeric variables.
#   df_select = df_cor %>% 
#     select_if(is.numeric) %>% 
#     select(variable, everything())
#   
#   # Calculate correlations and round to three decimal places.
#   corr = round(cor(df_select), 3)
#   
#   # Set lower triangle and diagonal values to NA to remove duplicates.
#   corr[lower.tri(corr, diag = TRUE)] = NA
#   
#   # Remove perfect correlations.
#   corr[corr == 1] = NA
#   
#   # Convert correlations to a 3-column data frame.
#   corr_df = reshape2::melt(corr[,1, drop = FALSE], na.rm=TRUE,
#                            variable.name = "Var1", value.name = "Freq") %>%
#     mutate(Freq = as.numeric(as.character(Freq)))
#   
#   # Remove rows with NA values.
#   corr_df = na.omit(corr_df)
#   
#   # Select correlations with absolute value greater 
#   # than the specified significance level.
#   corr_df = subset(corr_df, abs(Freq) > sig_level)
#   
#   # Sort correlations by absolute value in descending order.
#   corr_df = corr_df[order(-abs(corr_df$Freq)),]
#   
#   # Print the correlation table.
#   print(corr_df)
#   
#   # Convert correlation data frame to a matrix for visualization.
#   corr_matrix = reshape2::acast(corr_df, Var1 ~ Var2, value.var = "Freq")
#   
#   # Create a correlation plot.
#   corrplot(corr_matrix, method = "square", tl.col = "black", 
#            tl.srt = 45, is.corr = FALSE, tl.cex = 0.8, 
#            cl.cex = 0.8 , na.label = " ",  addCoef.col = "black")
#   
# }
# 
# # calculate_correlations_2(data = economics,
# #                          variable = "Cluster", 
# #                          sig_level = 0.7)
# # Results: No correlations found.
# 
# # Find the correlations between the economic variables
# # and the target cluster variable.
# economics_cor = sapply(economics[1:29], cor, 
#                        y = economics$Cluster, method = "spearman")
# 
# # Find the top correlated variables.
# top_vars = names(sort(abs(economics_cor), decreasing = TRUE))[1:10]
# top_vars = c("Cluster", top_vars[top_vars != "Cluster"])
# 
# # Create a correlation matrix for the top variables.
# corr_matrix = cor(economics[, top_vars], method = "spearman")
# 
# # Create a correlation plot for the top variables.
# corrplot(corr_matrix, method = "square", tl.col = "black", 
#          tl.srt = 45, is.corr = FALSE, tl.cex = 0.8, 
#          cl.cex = 0.8 , na.label = " ", addCoef.col = "black")
# 
# # Results:
# # > top_vars
# # [1] "Cluster"   "INC910213" "HSG495213" "INC110213"
# # [5] "BPS030214" "BZA010213" "SBO001207" "NES010213"
# # [9] "RTN130207" "BZA110213" "AFN120207"
# 
# # We will keep the highest correlated when two variables refer to similar 
# # economics measures, thus we will describe the clusters based 
# # on the following economics variables:
# # "INC910213" "HSG495213" "BPS030214" "BZA010213" 
# # "SBO001207" "NES010213" "RTN130207" "AFN120207"
################################################################################

# Add cluster assignments to the economics dataset.
economics_explanation = cbind(economics, Cluster = kmeans_7$cluster)

# Check the structure and summary statistics of the new dataset.
str(economics_explanation)
summary(economics_explanation)

# Compute the mean values of each variable for each cluster.
explanation_data = aggregate(economics_explanation, 
                             by = list(Cluster = economics_explanation$Cluster),
                             FUN = mean)

# Check the structure of the new dataset.
str(explanation_data)

# Remove the Cluster column from the explanation data.
explanation_data = subset(explanation_data, select = -c(Cluster))

# Load the GGally and plotly packages.
library(GGally) 
library(plotly)

# Create a parallel coordinate plot of the economics
# data with color-coded clusters.
p = ggparcoord(data = explanation_data, columns = c(1:29), 
               mapping = aes(color = as.factor(Cluster)),
               groupColumn = "Cluster", scale = "std") + 
  labs(x = "Economics Variables", y = "Cluster Values (std units)",
       title = "Economics Explanation Plot") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) 

# Convert the plot to an interactive plot using plotly 
# (saved also on drive as html).
ggplotly(p)

# Create a table to compare the mean values of each variable for each cluster.
table_data = as.data.frame(t(explanation_data))
colnames(table_data) = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4",
                         "Cluster 5", "Cluster 6", "Cluster 7")
print(table_data)

############################## END: Profiling ##################################

########################## END: Clustering Part II #############################