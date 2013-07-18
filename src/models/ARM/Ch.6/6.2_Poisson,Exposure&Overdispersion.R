library(rstan)
## Read in the data

#FIXME:missing data files?

## Model fitting

 # constant term
 #  glm (stops ~ 1, family=poisson, offset=log(arrests))

 # ethnicity indicator
 # glm (stops ~ factor(eth), family=poisson, offset=log(arrests))

 # ethnicity & precints indicators
 # glm (stops ~ factor(eth) + factor(precints) , family=poisson, offset=log(arrests))

 # overdispersion
 # glm (stops ~ factor(eth) + factor(precints) , family=quasipoisson, offset=log(arrests))

