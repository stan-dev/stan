/**
 * Command.stan used by src/test/models/command.stan to test arguments passed to
 * the compiled model.
 *
 * This is a copy of the src/models/basic_distributions/normal.stan model.
 */
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
