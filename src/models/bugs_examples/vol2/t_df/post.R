
library(coda) 
post1 <- read.csv(file = "samples1.csv", header = TRUE, comment.char = '#'); 
summary(as.mcmc(post1[, 'y.1'])) 

post2 <- read.csv(file = "samples2.csv", header = TRUE, comment.char = '#'); 
summary(as.mcmc(post2[, 'd'])) 

