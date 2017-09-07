parameters {
#include incl_nested.stan
}
transformed parameters {
  real w = y + z;
}
