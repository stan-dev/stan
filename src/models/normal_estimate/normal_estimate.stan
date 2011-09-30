data {
     int(0,) N;
     int(0,) M;
     double y[M,N];
}
parameters {
     double mu;
     double(0,) sigma;
}
model {
    for (m in 1:M) {
        for (n in 1:N) {
	    y[m,n] ~ normal(mu,sigma);
        }
    }
}

