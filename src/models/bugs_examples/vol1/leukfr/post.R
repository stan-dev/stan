## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 


pars <- c("beta", "sigma"); 
poi <- post[, pars]; 

summary(as.mcmc(poi)) 

library(BUGSExamples);
ex <- list(name = "Leukfr", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);
