library(coda);

## read samples from models specified 
## indivially (i.e., results from magnesium1.stan
## magnesium2.stan, ..., magnesium6.stan) 

post <- read.csv(file = 'samples1.csv', header = TRUE, comment.char = '#')
poi1 <- cbind(exp(post[, 1]), sqrt(post[, 2]));
colnames(poi1) <- c("OR1", "sigma1") 

post <- read.csv(file = 'samples2.csv', header = TRUE, comment.char = '#')
poi2 <- cbind(exp(post[, 1]), sqrt(post[, 2]));
colnames(poi2) <- c("OR2", "sigma2") 

post <- read.csv(file = 'samples3.csv', header = TRUE, comment.char = '#')
poi3 <- cbind(exp(post[, 1]), post[, 2]);
colnames(poi3) <- c("OR3", "sigma3") 

s0_sqrd <- 0.1272041; 
post <- read.csv(file = 'samples4.csv', header = TRUE, comment.char = '#')
poi4 <- cbind(exp(post[, 1]), sqrt(s0_sqrd * (1 / post[, 2] - 1)));
colnames(poi4) <- c("OR4", "sigma4") 

s0_sqrd <- 0.1272041; 
post <- read.csv(file = 'samples5.csv', header = TRUE, comment.char = '#')
poi5 <- cbind(exp(post[, 1]), sqrt(s0_sqrd) * (1 / post[, 2] - 1));
colnames(poi5) <- c("OR5", "sigma5") 

post <- read.csv(file = 'samples6.csv', header = TRUE, comment.char = '#')
poi6 <- cbind(exp(post[, 1]), sqrt(post[, 2]));
colnames(poi6) <- c("OR6", "sigma6") 

poi <-cbind(poi1, poi2, poi3, poi4, poi5, poi6); 
yapoi <- poi[, c(1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12)];
summary(as.mcmc(yapoi))
plot(as.mcmc(yapoi))


library(BUGSExamples)

pars <- c("OR", "tau"); 
ex <- list(name = "Magnesium", parameters = pars, 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 
plot(jagspost$coda);
