data {
    int N;
    int M;
    double z[M];
    double y[N];
    double mu;
    double(0,) sigma;
}
derived data {
}
parameters {
    int a;
    int b[5];
    int c[5,6,7];
}
derived parameters {
}
model {
   for (m in 1:M)
       z[m] ~ normal(mu,sigma);
   for (n in 1:N) 
       y[n] ~ normal(mu,sigma);
}
