data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
  real x;
}
transformed data {
  int td_i1 = 1 ? N : N;
  int td_i2 = 1 ? N : y[1];
  int td_i3 = 1 ? N : -3;
  int td_i4 = 1 ? 1 : 4;
  real td_r1 = 1 ? 1.0 : 2.0;
  real td_r2 = 1 ? 1.0 : x;
  real td_r3 = 1 ? x : 2.0;
  real td_r4 = 0 ? x : x;
  real td_r5 = 0 ? N : x;
}
parameters {
  real<lower=0,upper=1> theta;
  real z;
}
transformed parameters {
  real tp_r1 = 1 ? 1.0 : 2.0;
  real tp_r2 = 1 ? 1.0 : x;
  real tp_r3 = 1 ? x : 2.0;
  real tp_r4 = 0 ? x : x;
  real tp_r5 = 0 ? N : x;

  real tp_r6 = 1 ? x : tp_r1;
  real tp_r7 = 1 ? tp_r1 : tp_r2;
  real tp_r8 = 1 ? N : tp_r1;

  {
    real local_r1 = 1 ? 1.0 : 2.0;
    real local_r2 = 1 ? 1.0 : x;
    real local_r3 = 1 ? x : 2.0;
    real local_r4 = 0 ? x : x;
    real local_r5 = 0 ? N : x;

    real local_r6 = 1 ? x : tp_r1;
    real local_r7 = 1 ? tp_r1 : tp_r2;
    real local_r8 = 1 ? N : tp_r1;
  }
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  int gq_i1 = 1 ? N : N;
  int gq_i2 = 1 ? N : y[1];
  int gq_i3 = 1 ? N : -3;
  int gq_i4 = 1 ? 1 : 4;

  real gq_r1 = 1 ? 1.0 : 2.0;
  real gq_r2 = 1 ? 1.0 : x;
  real gq_r3 = 1 ? x : 2.0;
  real gq_r4 = 0 ? x : x;
  real gq_r5 = 0 ? N : x;

  real gq_r6 = 1 ? x : tp_r1;
  real gq_r7 = 1 ? tp_r1 : tp_r2;
  real gq_r8 = 1 ? N : tp_r1;

  {
    real local_r1 = 1 ? 1.0 : 2.0;
    real local_r2 = 1 ? 1.0 : x;
    real local_r3 = 1 ? x : 2.0;
    real local_r4 = 0 ? x : x;
    real local_r5 = 0 ? N : x;
    
    real local_r6 = 1 ? x : tp_r1;
    real local_r7 = 1 ? tp_r1 : tp_r2;
    real local_r8 = 1 ? N : tp_r1;
  }

}

