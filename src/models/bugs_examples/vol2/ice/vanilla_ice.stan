# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
# Page 29: Ice: non-parametric smoothing in an age-cohort model

data {
  int(0,) N; 
  int(0,) Nage;
  int(0,) None; 
  int(0,) K; 
  int year[N]; 
  int cases[N]; 
  int age[N]; 
  int pyr[N]; 
}
transformed data { # deterministic functions of data to be called only once
    int Nneighs[K];
    double shape;

    Nneighs[1] <- 1;
    Nneighs[2] <- 5;
    for (k in 3:(K-2))
        Nneighs[k] <- 6;
    Nneighs[K - 1] <- 5;
    Nneighs[K] <- 1;

    shape <- 0.0001 + K / 2.0;
}
parameters {
  double alpha[Nage - 1]; 
  double beta[K]; 
  double(0,) tau;
} 
transformed parameters { # deterministic functions of parameters called every iteration
    double betamean[K];
    double betasd[K];
    double tau_like[K];
    double log_RR[K];
    double d;

    betamean[1] <- 2.0 * beta[2] - beta[3];
    betamean[2] <- (2.0 * beta[1] + 4.0 * beta[3] - beta[4]) / 5.0;
    for (k in 3:(K-2)) {
        betamean[k] <- (4.0 * beta[k - 1] + 4.0 * beta[k + 1]- beta[k - 2] - beta[k + 2]) / 6.0;
    }
    betamean[K - 1] <- (2.0 * beta[K] + 4.0 * beta[K - 2] - beta[K - 3]) / 5.0;
    betamean[K] <- 2.0 * beta[K - 1] - beta[K - 2];
    for (k in 1 : K) {
        betasd[k] <- 1.0 / sqrt(Nneighs[k] * tau);
        log_RR[k] <- beta[k] - beta[5];
        tau_like[k] <- Nneighs[k] * beta[k] * (beta[k] - betamean[k]);
    }
    d <- 0.0001 + sum(tau_like) / 2.0;
}
model {
    double ln_mu;
    for(j in 1:(Nage-1)) {
        alpha[j] ~ normal(0.0, 1000.0);
    }
    tau ~ gamma(shape, d);

    for(i in 1:None) {
        ln_mu <- log(pyr[i]) + beta[year[i]];
        cases[i] ~ poisson(exp(ln_mu));
    }
    for(i in (None+1):N) {
        ln_mu <- log(pyr[i]) + alpha[age[i] - 1] + beta[year[i]];
        cases[i] ~ poisson(exp(ln_mu));
    }
}
