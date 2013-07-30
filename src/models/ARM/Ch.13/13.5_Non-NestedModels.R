library(rstan)
library(ggplot2)

## Read the pilots data & define variables FIXME
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/pilots

pilots <- read.table ("pilots.dat", header=TRUE)
attach.all (pilots)
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

## Model fit
## M1 <- lmer (y ~ 1 + (1 | group.id) + (1 | scenario.id))



## Plot figure 13.8




########################################################################################################
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/earnings

# The R codes & data files should be saved in the same directory for
# the source command to work

source("13.4_Understanding correlations between intercepts & slopes.R") # where data was cleaned

## Regression centering the predictors
##M1 <- lmer (y ~ x.centered + (1 + x.centered | eth) + (1 + x.centered | age) + (1 + x.centered | eth:age))
x.centered <- x - mean(x)
x.centered.jitter <- x.jitter - mean(x)


 # plot figure 13.10 ???????????













