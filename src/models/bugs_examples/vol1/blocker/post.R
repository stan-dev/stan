## take a look at the samples and compare with results computed 
## in other program. 


library(coda) 

N <- 22; 

post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 
colnames(post) <- c("d", "sigmasq_delta", 
                    paste("mu", 1:N, sep = ''), 
                    paste("delta", 1:N, sep = '')); 

sigma_delta <- sqrt(post[, "sigmasq_delta"]); 
delta.new <- post[, "d"] + rt(nrow(post), df = 4) * sigma_delta; 

poi <- cbind(post[, "d"], delta.new, sigma_delta); 
pars <- c("d", "delta.new", "sigma"); 
colnames(poi) <- pars; 
summary(as.mcmc(poi)) 

library(BUGSExamples)


# theta0 <- theta[1] - theta[2] * mean(X) 
ex <- list(name = "Blockers", parameters = pars, 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3); 

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 





