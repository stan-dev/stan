
library(coda)
post <- read.csv(file = "samples.csv", header = FALSE)[, 1:3]; 
colnames(post) <- c("alpha", "sigma", "theta"); 
summary(as.mcmc(post)) 
