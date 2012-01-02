# 
library(coda)
I <- 4
J <- 2
K <- 5
post <- read.csv(file = "samples2.csv", header = FALSE);

tmpfun <- function(x, y) {
  paste("[", x, ", ", y, "]", sep = '');
} 

# not sure how the parameters are ordered for 
# 2+ arrays, say, yabeta[I, K]. ?? 
colnames(post) <- c(paste("alpha", 2:K, sep = ''), 
                    paste("beta", as.vector(t(outer(2:I, 2:K, FUN = tmpfun))), sep = ''), 
                    paste("gamma", as.vector(t(outer(2:J, 2:K, FUN = tmpfun))), sep = ''), 
                    paste("lambda", as.vector(t(outer(1:I, 1:J, FUN = tmpfun))), sep = ''), 
                    paste("yaalpha", 1:K, sep = ''), 
#                   paste("yabeta", as.vector(t(outer(1:I, 1:K, FUN = tmpfun))), sep = ''),
#                   paste("yagamma", as.vector(t(outer(1:J, 1:K, FUN = tmpfun))), sep = ''));   
## for some reason, the derived parameters are in different order 
## [Mon Jan  2 13:18:29 EST 2012]
                    paste("yabeta", as.vector((outer(1:I, 1:K, FUN = tmpfun))), sep = ''),
                    paste("yagamma", as.vector((outer(1:J, 1:K, FUN = tmpfun))), sep = ''));   

summary(as.mcmc(post)) 


library(BUGSExamples);
pars <- c("alpha", "beta", "gamma"); 
ex <- list(name = "Alligators", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);


