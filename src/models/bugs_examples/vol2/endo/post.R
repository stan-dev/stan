library(coda)
post <- read.csv(file = 'samples1.csv', header = TRUE, comment.char = '#')
poi <- post[, 'beta'] 

summary(as.mcmc(poi)); 
quit('no')


library(BUGSExamples);
pars <- c("beta"); 
ex <- list(name = "Endo", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);
