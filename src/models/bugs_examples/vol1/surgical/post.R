library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, -(1:3)]
summary(as.mcmc(post)) 

# run in JAGS, BUGSExamples is needed.  
# BUGSExamples package is installed by 
# install.packages("BUGSExamples", repos = "http://R-Forge.R-project.org")

if (!is.element("BUGSExamples", installed.packages()[, 1])) {
    cat("Package BUGSExamples is not installed. It is used to get", 
        "the results from JAGS.", sep = '\n')
    quit();
} 


## For some reason, the package BUGSExamples could not 
## run this example. 

library(BUGSExamples)
# the simple model 
ex <- list(name = "Surgical", parameters = c("p"), 
           nSample = 2000, nBurnin = 500, nThin = 1, 
           nChain = 3)
 
jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 


ex <- list(name = "Surgicalrand", parameters = c("p", "mu", "sigma", "pop.mean"), 
           nSample = 2000, nBurnin = 500, nThin = 1, 
           nChain = 3)
 
jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 
