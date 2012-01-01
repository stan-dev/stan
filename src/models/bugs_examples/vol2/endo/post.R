library(coda)
post <- read.csv(file = 'samples3.csv', header = FALSE)
names(post)[1] <- c("beta");

summary(as.mcmc(post)); 


library(BUGSExamples);
pars <- c("beta"); 
ex <- list(name = "Endo", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);
