library(coda);
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#'); 

poi <- post[, c("alpha0", "alpha1", "alpha12", "alpha2", "sigma")] 
summary(as.mcmc(poi)) 
