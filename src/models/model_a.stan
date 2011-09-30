data {
  int I;
  double(3,5) z;
  double(0,1) y[(2*3)[1]];
  double x[2,3];
  corr_matrix(4) Sigma;
}
parameters {
  double(0,) sigma;
  double mu;
  double beta[3];
  double gamma[5,7];
  cov_matrix(4) Omega;
}
derived {
  int z;
  simplex(4) theta;
}
model {
    for (n in 1:N) {
        for (m in 1:M) {
	    z <- (u * w)[n];
            y[m,n] ~ Normal(mu[j],(x[n,m] * beta)[j][k]);
        }
    }
    mu ~ Normal(0,10);
    sigma ~ Cauchy(1.5);
}

