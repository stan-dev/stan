parameters {
  real mu;
  real<lower=0> sigma;
  real<offset=mu, multiplier=sigma> x;
  simplex[3] s;
  vector<multiplier=sigma>[2] v;
  tuple(real, real<multiplier=sigma>) t;
  real a;
  real<offset=a, multiplier=0.5> b;
  real<offset=b, multiplier=0.5> c;
  real<offset=c, multiplier=0.5> d;
  real lb;
  real<lower=lb> y;
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  x ~ normal(mu, sigma);
  v ~ normal(0, sigma);
  t.1 ~ normal(0, 1);
  t.2 ~ normal(0, sigma);
  a ~ normal(0, 1);
  b ~ normal(a, 0.5);
  c ~ normal(b, 0.5);
  d ~ normal(c, 0.5);
  lb ~ normal(0, 1);
  y ~ normal(lb + 1, 1);
}
