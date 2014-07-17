library(rstan)
library(ggplot2)

#############################################################################################
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/pilots

# The R codes & data files should be saved in the same directory for
# the source command to work

## Read the pilots data & define variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/pilots

pilots <- read.table ("pilots.dat", header=TRUE)
attach (pilots)
group.names <- as.vector(unique(group))
scenario.names <- as.vector(unique(scenario))
n.group <- length(group.names)
n.scenario <- length(scenario.names)
successes <- NULL
failures <- NULL
group.id <- NULL
scenario.id <- NULL
for (j in 1:n.group){
  for (k in 1:n.scenario){
    ok <- group==group.names[j] & scenario==scenario.names[k]    
    successes <- c (successes, sum(recovered[ok]==1,na.rm=T))
    failures <- c (failures, sum(recovered[ok]==0,na.rm=T))
    group.id <- c (group.id, j)
    scenario.id <- c (scenario.id, k)
  }
}

y <- successes/(successes+failures)
y.mat <- matrix (y, n.scenario, n.group)
sort.group <- order(apply(y.mat,2,mean))
sort.scenario <- order(apply(y.mat,1,mean))

group.id.new <- sort.group[group.id]
scenario.id.new <- sort.scenario[scenario.id]
y.mat.new <- y.mat[sort.scenario,sort.group]

scenario.abbr <- c("Nagoya", "B'ham", "Detroit", "Ptsbgh", "Roseln", "Chrlt", "Shemya", "Toledo")

## Define variables
treatment <- group.id
airport <- scenario.id

## Fit the 2-model using Bugs
n.treatment <- max(treatment)
n.airport <- max(airport)
n <- length(y)

## Model fit

dataList.2 <- list(N=n,y=y,n_airport=n.airport,
                   n_treatment=n.treatment,airport=airport,
                   treatment=treatment)
pilots_expansion.sf1 <- stan(file='pilots_expansion.stan', data=dataList.2,
                             iter=1000, chains=4)
print(pilots_expansion.sf1,pars = c("g","d", "sigma_y", "lp__"))

## Multilevel logistic regression
## radon model
polls.subset <- read.table ("polls.subset.dat")
attach(polls.subset)

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

 # define other data summaries
ok <- !is.na(bush)
y <- bush[ok]                  # 1 if support bush, 0 if support dukakis
n <- length(y)             # of survey respondents
n.age <- max(age)          # of age categories
n.edu <- max(edu)          # of education categories
n.state <- max(state)      # of states
n.region <- max(region)    # of regions
age.ok <- age[ok]
edu.ok <- edu[ok]
state.ok <- state[ok]
region.ok <- region[ok]
female.ok <- female[ok]
black.ok <- black[ok]

 # also include a measure of previous vote as a state-level predictor

library (foreign)
presvote <- read.dta ("presvote.dta")
attach (presvote)
v.prev <- presvote$g76_84pr
age.edu <- n.edu*(age-1) + edu
age.edu.ok <- age.edu[ok]
n.age.edu <- 16

dataList.2 <- list(N=n, n_age=n.age,n_edu=n.edu,n_state=n.state,
                   n_region=n.region, n_age_edu=n.age.edu,
                   y=y,female=female.ok,black=black.ok,
                   age=age.ok,edu=edu.ok, state=state.ok,region=region,
                   v_prev=v.prev,age_edu=age.edu.ok)
election88_expansion.sf1 <- stan(file='election88_expansion.stan',
                                 data=dataList.2, iter=1000, chains=4)
print(election88_expansion.sf1, pars = c("beta","b_age", "b_edu","b_state","b_region","b_age_edu", "lp__"))


## item response model -- FIXME: DATA??
