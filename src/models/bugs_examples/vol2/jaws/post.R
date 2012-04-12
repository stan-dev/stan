library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#') 
summary(as.mcmc(post)) 
plot(as.mcmc(post)) 

beginwith <- function(vstr, patterns, sort = TRUE) {
  patterns <- paste(patterns, ".*", sep = ''); 
  a <- lapply(patterns, FUN = function(p) {grep(p, vstr)}) 
  if (!sort)  return(do.call("c", a)) 
  sort(do.call("c", a))
} 

sigmas <- post[, beginwith(colnames(post), "Sigma\\.")]; 
invSigma <- t(apply(sigmas, 1, FUN = function(x) solve(matrix(x, ncol = 4)))) 
colnames(invSigma) <- gsub("Sigma", "invSigma", colnames(sigmas)) 
summary(as.mcmc(invSigma)) 


library(BUGSExamples);
# Sigma <- inverse(Omega) 
pars <- c("beta0", "beta1", "mu", "Omega", "Sigma"); 
ex <- list(name = "Jaws", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)



jagspost <- runExample(ex, engine = 'JAGS')

# apply(jagspost, ); 

summary(jagspost$coda); 




