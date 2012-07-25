# options(error = recover)
library(rstan)

scode <- "
parameters {
  real y[6, 7];
} 
model {
  for (i in 1:6) { 
    for (j in 1:7) { 
      y[i, j] ~ normal(0, 1); 
    }
  }
} 
"

s.f1 <- stan(model.code = scode, verbose = TRUE, n.chains = 1) 
print(s.f1)
plot(s.f1)

