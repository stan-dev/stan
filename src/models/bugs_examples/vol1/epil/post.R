
library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#') 

pars <- c("alpha_Age", "alpha_BT", "alpha_Base", "alpha_Trt", "alpha_V4", "alpha0", "sigma_b", "sigma_b1"); 

poi <- post[, pars]  
summary(as.mcmc(poi)) 


library(BUGSExamples);
pars2 <- c("alpha.Age", "alpha.BT", "alpha.Base", "alpha.Trt", "alpha.V4", "alpha0", "sigma.b", "sigma.b1"); 
ex <- list(name = "Epil", parameters = pars2,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda); 


