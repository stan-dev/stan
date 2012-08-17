library(rstan)

model_name <- "_8chools";
sfile <- "../../src/models/misc/eight_schools/eight_schools.stan"
m <- stan.model(file = sfile, 
                model.name = model_name, 
                verbose = TRUE)  

J <- 8L 
y <- c(28,  8, -3,  7, -1,  1, 18, 12)
sigma <- c(15, 10, 16, 11,  9, 11, 10, 18)

n.iter <- 1000
# specify data using names 
ss1 <- sampling(m, data = c("J", "y", "sigma"), n.iter = n.iter, n.chains = 4, equal_step_sizes = FALSE, refresh = 100) 

print(ss1) 
traceplot(ss1)

dat <- c("J", "y", "sigma") 
ss <- stan(sfile, data = dat, n.iter = n.iter, n.chains = 4, sample.file = '8schools.csv')
print(ss)
plot(ss) 


# using previous fitted objects 
ss2 <- stan(fit = ss, data = dat, n.iter = 2000) 
print(ss2, probs = c(0.38))
print(ss2, probs = c(0.48))
