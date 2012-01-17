library(coda); 

## read samples from models specified all-together (magnesium_all.stan)

post <- read.csv(file = 'samples.csv', header = TRUE)
colnames(post)[1:6] <- paste('mu', 1:6, sep = ''); 

s0_sqrd <- 0.1272041; 
poi1 <- exp(post[, 1:6]); 
pois1 <- sqrt(post[, 7]); 
pois2 <- sqrt(post[, 8]); 
pois3 <- post[, 9]; 
pois4 <- sqrt(s0_sqrd * (1 / post[, 10] - 1)); 
pois5 <- sqrt(s0_sqrd) * (1 / post[, 11] - 1); 
pois6 <- sqrt(post[, 12]); 

poi <- cbind(poi1, pois1, pois2, pois3, pois4, pois5, pois6); 
colnames(poi) <- c(paste("OR", 1:6, sep = ''), paste("sigma", 1:6, sep = '')); 
summary(as.mcmc(poi))
plot(as.mcmc(poi)) 
