transformed data {
    real L; 
    real H;
    L <- -5.0;
    H <- 5.0;
}
parameters {
    real[L,H] a;
    real[a,H] b;
}
model {
//    a ~ uniform(L,b);
//    b ~ uniform(a,H);
}
