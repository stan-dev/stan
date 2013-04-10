# add a print statement 

library(rstan)

school_code <- "
data {
    int<lower=0> J;           // number of schools
    real y[J];                // estimated treatment effect (school j)
    real<lower=0> sigma[J];   // std dev of effect estimate (school j)
}
parameters {
    real mu;
    real theta[J];
    real<lower=0> tau;
}
model {
    theta ~ normal(mu, tau); 
    print(theta); 
    y ~ normal(theta,sigma);
}
"

J <- 8L 
y <- c(28,  8, -3,  7, -1,  1, 18, 12)
sigma <- c(15, 10, 16, 11,  9, 11, 10, 18)

iter <- 1000
dat <- c("J", "y", "sigma") 
ssp <- stan(model_code = school_code, data = dat, iter = iter, chains = 4, refresh = -1) 
ssp2 <- stan(fit = ssp, data = dat) 
print(ssp)

gqtestcode <- '
parameters {
  real y;
}
model {
  y ~ normal(0,1);
  print("model y=",y);
}
generated quantities {
  real z;
  z <- 2 * y;
  print("gq z=",z);
}
'

gqfit <- stan(model_code = gqtestcode, chains = 1, iter = 10)




