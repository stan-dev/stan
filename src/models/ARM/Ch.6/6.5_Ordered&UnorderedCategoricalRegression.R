library(rstan)
library(ggplot2)
library(boot)
## Read in the data

##FIXME: MISSING data?

## Fitting the model
## fit.1 <- bayespolr (factor(y) ~x)


## Displaying the fitted model
expected <- function (x, c1.5, c2.5, sigma){
  p1.5 <- inv.logit ((x-c1.5)/sigma)
  p2.5 <- inv.logit ((x-c2.5)/sigma)
  return ((1*(1-p1.5) + 2*(p1.5-p2.5) + 3*p2.5))
}

## Plots
