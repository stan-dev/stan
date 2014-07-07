## The finite-population group-level coefficient

attach.bugs(AP.fit) ## you would need to replace this with an extraction of a model fit in stan
finite.slope <- rep (NA, n.sims)
for (s in 1:n.sims){
  finite.pop <- lm (a[s,] ~ u) ##you would need to replace this with a call to a stan model
  finite.slope[s] <- coef(finite.pop)["u"]
}
quantile (finite.pop, c(.025, .975))
