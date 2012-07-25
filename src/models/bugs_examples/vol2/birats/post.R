library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#') 

beginwith <- function(vstr, patterns) { 
  patterns <- paste(patterns, ".*", sep = ''); 
  a <- lapply(patterns, FUN = function(p) {grep(p, vstr)}) 
  do.call("c", a)
} 


sigma_beta <- post[, beginwith(colnames(post), "Sigma_beta\\.")]   
inv_sigma_beta <- t(apply(sigma_beta, 1, FUN = function(x) { m <- matrix(x, ncol = 2); as.vector(solve(m))}))
colnames(inv_sigma_beta) <- paste('inv_', colnames(sigma_beta), sep = '')

poi <- cbind(post[, c("mu_beta.1", "mu_beta.2", "sigma_y")], sigma_beta, inv_sigma_beta) 
summary(as.mcmc(poi)) 

library(BUGSExamples);
pars <- c("mu.beta", "sigma", "R"); 
ex <- list(name = "BiRats", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda); 




