library(coda); 

## read samples from models specified all-together (magnesium.stan)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')
poi.names <- c(paste("OR.", 1:6, sep = ''), paste("tau.", 1:6, sep = '')); 
poi <- post[, poi.names] 
summary(as.mcmc(poi))
plot(as.mcmc(poi)) 
