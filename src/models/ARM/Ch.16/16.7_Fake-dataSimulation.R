library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Call Stan

radon.2.sf <- stan(file='radon.2.stan', data=radon.data, iter=1000, chains=4)

## Specifying the unmodeled parameters

sims <- extract(radon.2.sf)
b.true <- median(sims$b)
g.0.true <- median(sims$g_0)
g.1.true <- median(sims$g_1)
sigma.y.true <- median(sims$sigma_y)
sigma.a.true <- median(sims$sigma_a)

## Simulating the varying coefficients
a.true <- rep (NA, J)
for (j in 1:J){
  a.true[j] <- rnorm(1, g.0.true + g.1.true * u[j], sigma.a.true)
}

## Simulating fake data
y.fake <- rep (NA, N)
for (i in 1:N){
  y.fake[i] <- rnorm (1, a.true[county[i]] + b.true*x[i], sigma.y.true)
}

## Inference and comparison to "true" values

# specify the data
radon.data <- list(N = N, J = J, y = y.fake, county = county, x = x, u = u)

# call Stan
radon.2.fake.sf <- stan(file='radon.2.stan', data=radon.data, iter=1000,
                        chains=4)
print(radon.2.fake.sf)

## Checking coverage of 50% intervals
sims <- extract(radon.2.fake.sf)

# coverage for alpha1
a.true[1] > quantile(sims$a[1,], .25) & a.true[1] < quantile(sims$a[1,], .75)

# coverage for the 85 alphas
cover.50 <- rep(NA, J)
for (j in 1:J) {
  cover.50[j] <- a.true[j] > quantile(sims$a, .25) & 
                 a.true[j] < quantile(sims$a, .75)
}
mean(cover.50)
