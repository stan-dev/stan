/**
 * Command.stan used by src/test/models/command.stan to test arguments passed to
 * the compiled model.
 *
 */
data {
  real mu;
}
parameters {
  real y;
}
model {
  y ~ normal(mu,1);
}
