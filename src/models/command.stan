/**
 * Command.stan used by src/test/models/command.stan to test arguments passed to
 * the compiled model.
 */
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
