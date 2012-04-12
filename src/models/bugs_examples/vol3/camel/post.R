library(coda)
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#') 
 
plot(as.mcmc(post)) 

beginwith <- function(vstr, patterns, sort = TRUE) {
  patterns <- paste(patterns, ".*", sep = '');
  a <- lapply(patterns, FUN = function(p) {grep(p, vstr)})
  if (!sort)  return(do.call("c", a))
  sort(do.call("c", a))
}

poiidx <- beginwith(colnames(post), c("Sigma", "rho"))
poi <- post[, poiidx] 
summary(as.mcmc(poi)) 

plot(as.mcmc(poi)) 


## JAGS does not support the missing data here
## 
#   library(BUGSExamples);
#   pars <- c("mu", "Sigma2", "tau", "rho"); 
#   ex <- list(name = "Camel", parameters = pars,
#              nSample = 10000, nBurnin = 1000, nThin = 1,
#              nChain = 3)

#   jagspost <- runExample(ex, engine = "JAGS") 
#   summary(jagspost) 



