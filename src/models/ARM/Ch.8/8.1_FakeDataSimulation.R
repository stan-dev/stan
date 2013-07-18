## Fake-data simulation
library ("arm")

a <- 1.4
b <- 2.3
sigma <- 0.9
x <- 1:5
n <- length(x)

# Simulate data, fit the model, and check the coverage of the conf intervals
#FIXME switch to stan when sim sorted out
y <- a + b*x + rnorm (n, 0, sigma)
lm.1 <- lm (y ~ x)
display (lm.1)

b.hat <- coef (lm.1)[2]       # "b" is the 2nd coef in the model
b.se <- se.coef (lm.1)[2]     # "b" is the 2nd coef in the model

cover.68 <- abs (b - b.hat) < b.se     # this will be TRUE or FALSE
cover.95 <- abs (b - b.hat) < 2*b.se   # this will be TRUE or FALSE
cat (paste ("68% coverage: ", cover.68, "\n"))
cat (paste ("95% coverage: ", cover.95, "\n"))

# Put it in a loop

n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  lm.1 <- lm (y ~ x)
  b.hat <- coef (lm.1)[2]      
  b.se <- se.coef (lm.1)[2]   
  cover.68[s] <- abs (b - b.hat) < b.se  
  cover.95[s] <- abs (b - b.hat) < 2*b.se 
}
cat (paste ("68% coverage: ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))

# Do it again, this time using t intervals

n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
t.68 <-  qt (.84, n-2)
t.95 <-  qt (.975, n-2)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  lm.1 <- lm (y ~ x)
  b.hat <- coef (lm.1)[2]      
  b.se <- se.coef (lm.1)[2]   
  cover.68[s] <- abs (b - b.hat) < t.68*b.se  
  cover.95[s] <- abs (b - b.hat) < t.95*b.se 
}
cat (paste ("68% coverage: ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))  
