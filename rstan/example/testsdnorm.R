# options(error = recover)
library(rstan)

scode <- "
data {
  int I; 
  int J; 
} 
parameters {
  vector[I] y[J]; 
} 
model {
  for (j in 1:J)  y[j] ~ normal(0, 1); 
} 
"

sf1 <- stan(model_code = scode, verbose = TRUE, chains = 1, data = list(I = 6, J = 7)) 
# print(sf1)
# plot(sf1)

sf2 <- stan(fit = sf1, iter = 10000, chains = 4, warmup = 1000, data = list(I = 6, J = 7))
system.time(y3_2 <- extract(sf2, 'y[3,2]', inc_warmup = FALSE))
save.image(file = 'norms.RData')

dim(sf1) 
dimnames(sf1)
a <- as.array(sf1)
is.array(sf1)
m <- as.matrix(sf1)
e <- extract(sf1)
dimnames(e) 

sf3 <- stan(fit = sf1, test_grad = TRUE, data = list(I = 3, J = 4))
dimnames(sf3)
is.array(sf3)
a3 <- as.array(sf3)
m3 <- as.matrix(sf3)
dim(sf3) 


