library(coda) 
# library(BUGSExamples) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

poi <- post[, c(paste("alpha.", 1:5, sep = ''), paste('a.', 1:5, sep = ''), "beta")] 
poi <- as.mcmc(poi)
summary(poi) 

## compare with benchmark from JAGS example
benchstats <-
structure(c(-2.745633835, -1.00315582550000, -0.246628370775000, 
-1.3127207735, -2.1066172, 0.7594457955, 0.134599617817441, 0.077534970253042, 
0.0741992325942451, 0.084149591609884, 0.108905719330582, 0.0711087756187988, 
0.00300973895185289, 0.00173373464119226, 0.00165914527959050, 
0.00188164207118547, 0.00243520591561694, 0.00159004056080414, 
0.00455924426747034, 0.00290739461757728, 0.0026965506231021, 
0.0027963649898875, 0.00380594358630786, 0.00436433796205854), .Dim = as.integer(c(6, 
4)), .Dimnames = list(c("alpha[1]", "alpha[2]", "alpha[3]", "alpha[4]", 
"alpha[5]", "beta"), c("Mean", "SD", "Naive SE", "Time-series SE"
)))

print(benchstats) 


#   pars <- c("alpha", "a", "beta", "theta"); 
#   ex <- list(name = "Lsat", parameters = pars,
#              nSample = 10000, nBurnin = 1000, nThin = 1,
#              nChain = 3)

#   jagspost <- runExample(ex, engine = 'JAGS')
#   summary(jagspost$coda); 



