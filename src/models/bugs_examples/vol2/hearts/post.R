library(coda)
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#', skip = 19);

pars <- c("alpha", "delta", "theta"); 

summary(as.mcmc(post[, pars])) 

library(BUGSExamples);
ex <- list(name = "Hearts", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);

