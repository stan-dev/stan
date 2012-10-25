data {
  vector<lower=0>[2] v0;
  vector<upper=1>[2] v1;
  vector<lower=0,upper=1>[2] v01;

  row_vector<lower=0>[3] rv0;
  row_vector<upper=1>[3] rv1;
  row_vector<lower=0,upper=1>[3] rv01;

  matrix<lower=0>[2,3] m0;
  matrix<upper=1>[2,3] m1;
  matrix<lower=0,upper=1>[2,3] m01;
}
model {
  0 ~ bernoulli(0.5);
}
