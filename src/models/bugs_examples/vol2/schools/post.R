library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')

poiidx <- c(grep('beta\\.[0-9]+', colnames(post)), 
            grep('gamma\\.[0-9]+', colnames(post)), 
            grep('phi', colnames(post)), 
            grep('theta', colnames(post)))


poi <- post[, poiidx] 

summary(as.mcmc(poi)); 
# quit('no')

ranks <- post[, grep('ranks\\.[0-9]+', colnames(post))] 
ranksm <- apply(ranks, 2, mean) 
dotplot(rev(sort(ranksm)))




require(BUGSExamples);
pars <- c("beta", "gamma", "phi", "theta") 

ex <- list(name = "Schools", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
# plot(jagspost$coda);
