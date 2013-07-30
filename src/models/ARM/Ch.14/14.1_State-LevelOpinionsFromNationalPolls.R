library(rstan)
library(ggplot2)
## Read the data & define variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88

# Set up the data for the election88 example

# Load in data for region indicators
# Use "state", an R data file (type ?state from the R command window for info)
#
# Regions:  1=northeast, 2=south, 3=north central, 4=west, 5=d.c.
# We have to insert d.c. (it is the 9th "state" in alphabetical order)

data (state)                  # "state" is an R data file
state.abbr <- c (state.abb[1:8], "DC", state.abb[9:50])
dc <- 9
not.dc <- c(1:8,10:51)
region <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4)

# Load in data from the CBS polls in 1988
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88
library (foreign)
polls <- read.dta ("polls.dta")
attach.all (polls)

# Select just the data from the last survey (#9158)
table (survey)                # look at the survey id's
ok <- survey==9158            # define the condition
polls.subset <- polls[ok,]    # select the subset of interest
attach.all (polls.subset)     # attach the subset
write.table (polls.subset, "polls.subset.dat")

print (polls.subset[1:5,])

# define other data summaries
y <- bush                  # 1 if support bush, 0 if support dukakis
n <- length(y)             # of survey respondents
n.age <- max(age)          # of age categories
n.edu <- max(edu)          # of education categories
n.state <- max(state)      # of states
n.region <- max(region)    # of regions

# compute unweighted and weighted averages for the U.S.
ok <- !is.na(y)                                    # remove the undecideds
cat ("national mean of raw data:", round (mean(y[ok]==1), 3), "\n")
cat ("national weighted mean of raw data:",
     round (sum((weight*y)[ok])/sum(weight[ok]), 3), "\n")

# compute weighted averages for the states
raw.weighted <- rep (NA, n.state)
names (raw.weighted) <- state.abbr
for (i in 1:n.state){
  ok <- !is.na(y) & state==i
  raw.weighted[i] <- sum ((weight*y)[ok])/sum(weight[ok])
}

# load in 1988 election data as a validation check
election88 <- read.dta ("election88.dta")
outcome <- election88$electionresult

# load in 1988 census data
census <- read.dta ("census88.dta")

# also include a measure of previous vote as a state-level predictor
presvote <- read.dta ("presvote.dta")
attach (presvote)
v.prev <- presvote$g76_84pr
not.dc <- c(1:8,10:51)
candidate.effects <- read.table ("candidate_effects.dat", header=T)
v.prev[not.dc] <- v.prev[not.dc] +
 (candidate.effects$X76 + candidate.effects$X80 + candidate.effects$X84)/3
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88

## Multilevel logistic regression
## M1 <- lmer (y ~ black + female + (1 | state), family=binomial(link="logit"))

## A fuller model

 # set up the predictors
age.edu <- n.edu*(age-1) + edu
region.full <- region[state]
v.prev.full <- v.prev[state]

# fit the model
# M2 <- lmer (y ~ black + female + black:female + v.prev.full + (1 | age) + (1 | edu) + (1 | age.edu) + (1 | state) + (1 | region.full), family=binomial(link="logit"))

## Plot Figure 14.1 
