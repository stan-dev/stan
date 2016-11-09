parameters {
  real y[3];

}
transformed parameters {
  simplex[size(y)] z;
}
