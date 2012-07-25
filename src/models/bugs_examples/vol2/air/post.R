
library(coda) 
J <- 3; 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, -(1:3)] 
summary(as.mcmc(post)) 

# run in JAGS 
library(BUGSExamples)


# theta0 <- theta[1] - theta[2] * mean(X) 
ex <- list(name = "Air", parameters = c("theta[1]", "theta[2]", paste("X[", 1:J, "]", sep = '')), 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3); 

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 


# P.S. (from the Readme of the corresponding JAGS example)
# ----- 
# The posterior distributions of theta0 and theta have a very heavy tail.
# Excursions into the tail are rare, but have a strong influence on
# the mean.
# 
# theta0 <- theta[1] - theta[2] * mean(X) 

