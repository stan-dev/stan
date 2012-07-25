library(coda) 
library(BUGSExamples)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')[,-(1:3)]
summary(as.mcmc(post)) 

ex <- list(name = "Dyes", parameters = c("theta", "sigma2.with", "sigma2.btw"),
           nSample = 2000, nBurnin = 500, nThin = 1, 
           nChain = 3)
 
jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 
