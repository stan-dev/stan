library(coda);

## read samples from models specified 
## indivially (i.e., results from magnesium1.stan
## magnesium2.stan, ..., magnesium6.stan) 

post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')

library(BUGSExamples)

pars <- c("OR", "tau"); 
ex <- list(name = "Magnesium", parameters = pars, 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 
plot(jagspost$coda);
