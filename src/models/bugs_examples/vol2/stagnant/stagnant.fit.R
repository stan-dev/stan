library('rstan')
source('stagnant.data.R')
fit <- stan('stagnant2.stan',
            data=c("N","x","Y"),
            chains=4, iter=2000);
