data {
    int M;
    int N;
    double y[M];
}
derived data {
}
parameters {
    int a;
    int b[5];
    int c[5,6,7];
    double z[N];
    double(0,) sigma;
    double mu;
    vector(17) yy;
    vector(19) zz[3];
    matrix(2,3) aa;
    matrix(4,5) bb[6];
    matrix(7,8) cc[9,10,11];
}
derived parameters {
}
model {
   for (m in 1:M)
       y[m] ~ normal(mu,sigma);
   for (n in 1:N) 
       z[n] ~ normal(mu,sigma);
}
