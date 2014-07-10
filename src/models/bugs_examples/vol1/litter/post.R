## take a look at the samples and compare with results computed 
## in other program. 


library(coda) 

post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#');

summary(as.mcmc(poi)) 

library(BUGSExamples)


# theta0 <- theta[1] - theta[2] * mean(X) 
ex <- list(name = "Litter", parameters = pars, 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3); 

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 





