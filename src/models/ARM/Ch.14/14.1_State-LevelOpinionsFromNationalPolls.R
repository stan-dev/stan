library(rstan)
library(ggplot2)
library(gridBase)
library(boot)
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
attach(polls)

# Select just the data from the last survey (#9158)
table (survey)                # look at the survey id's
ok <- survey==9158            # define the condition
polls.subset <- polls[ok,]    # select the subset of interest
attach(polls.subset)     # attach the subset
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
attach(presvote)
v.prev <- presvote$g76_84pr
not.dc <- c(1:8,10:51)
candidate.effects <- read.table ("candidate_effects.dat", header=T)
v.prev[not.dc] <- v.prev[not.dc] +
 (candidate.effects$X76 + candidate.effects$X80 + candidate.effects$X84)/3
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88

## Multilevel logistic regression (election88.stan)
## M1 <- lmer (y ~ black + female + (1 | state), family=binomial(link="logit"))

ok <- !is.na(y+black+female+state)

dataList.1 <- list(N=length(y[ok]),y=y[ok],n_state=n.state,black=black[ok],female=female[ok],state=state[ok])
election88.sf1 <- stan(file='election88.stan', data=dataList.1,
                       iter=1000, chains=4)
print(election88.sf1, pars = c("a","b", "lp__"))

## A fuller model

 # set up the predictors
age.edu <- n.edu*(age-1) + edu
region.full <- region[state]
v.prev.full <- v.prev[state]
n.age.edu <- 16
n.region.full <- 5

ok <- !is.na(y+black+female+state+v.prev.full+age+edu+age.edu+region.full)

# fit the model
# M2 <- lmer (y ~ black + female + black:female + v.prev.full + (1 | age) + (1 | edu) + (1 | age.edu) + (1 | state) + (1 | region.full), family=binomial(link="logit"))

dataList.2 <- list(N=length(y[ok]),
                   y=y[ok],
                   black=black[ok],
                   female=female[ok],
                   age=age[ok],
                   state=state[ok],
                   age_edu=age.edu[ok],
                   region_full=region.full[ok],
                   edu=edu[ok],
                   n_age=n.age,
                   n_edu=n.edu,
                   n_state=n.state,
                   n_age_edu=n.age.edu,
                   n_region_full=n.region.full,
                   v_prev_full=v.prev.full[ok])
election88_full.sf1 <- stan(file='election88_full.stan', data=dataList.2,
                            iter=1000, chains=4)
print(election88_full.sf1, pars = c("beta","a","b", "c","d","e","lp__"))
post <- extract(election88_full.sf1)
beta <- colMeans(post$beta)
a.age <- colMeans(post$a)
a.edu <- colMeans(post$b)
a.age.edu <- colMeans(post$c)
a.state <- post$d
## Plot Figure 14.3???

## Plot Figure 14.2 

  # create linear predictors
linpred <- rep (NA, n)
for (i in 1:n){
  linpred[i] <- mean (beta[1] + beta[3]*female[i] + beta[2]*black[i] +
    beta[5]*female[i]*black[i] + a.age[age[i]] + a.edu[edu[i]] +
    a.age.edu[age.edu[i]])
}

  # plot the 8 states
dev.new()
state.name.all <- c(state.name[1:8], "District of Columbia", state.name[9:50])

pushViewport(viewport(layout = grid.layout(2, 4)))

p2 <- "ggplot() +
       geom_point() +
       geom_jitter(position = position_jitter(width = .05, height = 0.05)) +
        scale_x_continuous('Linear Predictor') +
        scale_y_continuous('Pr(support bush)') +
        labs(title=state.name.all[2]) +
        theme_bw()"
for (i in 1:20) {
  p2 <- paste(p2,"+ stat_function(aes(y=0),fun=function(x) 1.0 / (1 + exp(a.state[4000-",i,",2] - x)),colour='grey')")
}
  p2 <- paste(p2, "+ stat_function(fun=function(x) 1.0 / (1 + exp(colMeans(a.state)[2] - x)))")
  print(eval(parse(text = p2)), vp = viewport(layout.pos.row = 1, layout.pos.col = 1))

k <- 1
for (j in c(3,4,8,6,7,5,9)) {
  frame1 = data.frame(y1=y[state==j],x1=linpred[state==j])
  p2 <- "ggplot(frame1, aes(x=x1, y=y1)) +
        geom_jitter(position = position_jitter(width = .05, height = 0.05)) +
        scale_x_continuous('Linear Predictor') +
        scale_y_continuous('Pr(support bush)') +
        labs(title=state.name.all[j]) +
        theme_bw()"
for (i in 1:20) {
  p2 <- paste(p2,"+ stat_function(aes(y=0),fun=function(x) 1.0 / (1 + exp(a.state[4000-",i,",j] - x)),colour='grey')")
}
  p2 <- paste(p2, "+ stat_function(fun=function(x) 1.0 / (1 + exp(colMeans(a.state)[j] - x)))")
  print(eval(parse(text = p2)), vp = viewport(layout.pos.row = floor(k / 4) + 1, layout.pos.col = k - floor(k / 4) * 4+1))
  k <- k + 1
  }

## Using the model inferences to estimate avg opinion for each state

 # construct the n.sims x 3264 matrix
n.sims <- 4000
a.age <- (post$a)
a.edu <- (post$b)
a.age.edu <- (post$c)
a.state <- post$d
a.region <- post$e
L <- nrow (census)
y.pred <- array (NA, c(n.sims, L))
for (l in 1:L){
  y.pred[,l] <- inv.logit(beta[1] + beta[3] *census$female[l] +
    beta[2] *census$black[l] + beta[5]*census$female[l]*census$black[l] +
    a.age[,census$age[l]] + a.edu[,census$edu[l]] +
    a.age.edu[,n.edu*(census$age[l]-1) + census$edu[l]] + a.state[,census$state[l]])
}

 # average over strata within each state
y.pred.state <- array (NA, c(n.sims, n.state))
for (s in 1:n.sims){
  for (j in 1:n.state){
    ok <- census$state==j
    y.pred.state[s,j] <- sum(census$N[ok]*y.pred[s,ok])/sum(census$N[ok])
  }
}

 # average over strata within each state
state.pred <- array (NA, c(n.state,3))
for (j in 1:n.state){
  state.pred[j,] <- quantile (y.pred.state[,j], c(.25,.5,.75))
}

## Plot Figure 14.3

dev.new()
pushViewport(viewport(layout = grid.layout(1, 4)))
region.name <- c("Northeast", "Midwest", "South", "West", "D.C.")

for (k in 1:4) {
  p3 <- "ggplot() +
        geom_jitter(position = position_jitter(width = .05, height = .05)) +
        scale_x_continuous('R Vote in Previous Elections') +
        scale_y_continuous('Regression Intercepts', limits=c(-1,.7)) +
        labs(title=region.name[k]) +
        theme_bw()"
for (i in (1:n.state)[region==k]) {
  p3 <- paste(p3," + geom_segment(aes(x=v.prev[",i,"],y=quantile(a.state[,",i,"], c(.25,.75))[1],xend=v.prev[",i,"],yend=quantile(a.state[,",i,"], c(.25,.75))[2]),colour='grey')")
}
  p3 <- paste(p3," + geom_abline(intercept=median(a.region[,k]) - .7,slope=median(v.prev))")
  print(eval(parse(text = p3)), vp = viewport(layout.pos.row = 1, layout.pos.col = k))
  }
