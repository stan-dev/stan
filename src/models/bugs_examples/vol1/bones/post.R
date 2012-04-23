
library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

theta_idx <- grep("theta.[:digits:]*", colnames(post))
summary(as.mcmc(post[, theta_idx])) 

pars <- c('theta') 
library(BUGSExamples);
ex <- list(name = "Bones", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);
