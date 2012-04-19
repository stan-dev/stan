## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

sigma <- 1 / sqrt(post[, 2]) 

poi <- cbind(post[, 1], sigma); 
colnames(poi) <- c("beta", "sigma")

summary(as.mcmc(poi)) 
