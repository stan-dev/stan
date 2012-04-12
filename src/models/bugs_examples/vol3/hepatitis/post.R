library(coda) 
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')


pars <- c("gamma", "alpha0", "beta0", "sigma_y", "sigma_alpha", "sigma_beta"); 
poi <- post[, pars] 
summary(as.mcmc(poi)) 


## for some reason, it does not work here to use JAGS 
## run this example. 
#   library(BUGSExamples);
#   pars <- c("gamma", "alpha0", "beta0", "sigma"); 
#   ex <- list(name = "HepatitisME", parameters = pars,
#              nSample = 10000, nBurnin = 1000, nThin = 1,
#              nChain = 3)
#                                                          
#   jagspost <- runExample(ex, engine = "JAGS") 
#   summary(jagspost) 
