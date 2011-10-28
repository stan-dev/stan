# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 3: Rats
data {
    int N;
    int T;
    double x[T];
    double xbar;
}
parameters {
    double mu[N,T];
    double(0,) tau_c;
    double alpha[N];
    double beta[N];
    double alpha_c;
    double alpha_tau;
    double beta_c;
    double beta_tau;
    double(0,) Y[N,T];
    double sigma;
    double alpha0;
}
model {
    for (i in 1:N) {
       for (j in 1:T) {
             Y[i, j] ~ normal(mu[i , j], tau_c);
             mu[i, j] <- alpha[i] + beta[i] * (x[j] - xbar);
         }
         alpha[i] ~ normal(alpha_c, alpha_tau);
         beta[i] ~ normal(beta_c, beta_tau);
     } 
     tau_c ~ gamma(0.001,0.001);
     sigma <- 1 / sqrt(tau_c);
     alpha_c ~ normal(0.0,1.0E-6);
     alpha_tau ~ gamma(0.001,0.001);
     beta_c ~ normal(0.0,1.0E-6);
     beta_tau ~ gamma(0.001,0.001);
     alpha0 <- alpha_c - xbar * beta_c;
}
