transformed data {
    real L; 
    real H;
    L <- -5.0;
    H <- 5.0;
}
parameters {
    real<lower=L,upper=H> a;
    real<lower=a,upper=H> b;
}
model {
//    a ~ uniform(L,b);
//    b ~ uniform(a,H);
}
