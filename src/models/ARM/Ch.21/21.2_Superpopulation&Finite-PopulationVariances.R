## Finite-population sd for regression coefficients that are not part of
## a multilevel model

b.mean <- sum (b*lambda)
b.sd <- sqrt (sum (lambda*(b-b.mean)^2))

 # or
b.sd <- sd (b[x])
