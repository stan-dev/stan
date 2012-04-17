data {
    int(1,) K;
    int(1,) N;
    real y[N];
}
parameters {
    simplex(K) theta;
    real mu[K];
    real sigma[K];
}
model {
    for (k in 1:K) {
        mu[k] ~ normal(0,10);
        sigma[k] ~ gamma(1,1);
    }
    for (n in 1:N) {
        real log_ps[K];
        for (k in 1:K)
           log_ps[k] <- log(theta[k]) + normal_log(y[n],mu[k],sigma[k]);
        lp__ <- lp__ + log_sum_exp(log_ps);
    }
}

// EFFICIENCY
// transformed paramters {
//    real log_theta[K];
//    for (k in 1:K)
//        log_theta[k] <- log(theta[k]);

// RESPONSIBILITY
// derived quantities {
//    matrix(N,K) resp;
//    for (n in 1:N) {
//        for (k in 1:K) 
//            resp[n,k] <- theta[k] * exp(normal_log(y[n],mu[k],sigma[k]));
//        resp[n] <- resp[n] / sum(resp[n]);
//   }
// }