
# read the simulated student t samples 
library(coda)
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, "y.1"]; 
colnames(post) <- c("alpha", "sigma", "theta"); 
summary(as.mcmc(post)) 
