functions {
  vector foo(int a1) {
    vector[a1] lf1;
    vector[a1] lf2 = lf1;
    vector[a1] lf3[a1];
    vector[a1] lf4[a1] = lf3;
    print("foo, a1: ",a1);
    print("foo, lf1: ", lf1);
    print("foo, lf2: ", lf1);
    print("foo, lf4: ", lf4);
    return lf1;
  }
}
data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  int a = 1;
  vector[a] f1;
  vector[a] bar = f1;
  row_vector[a] baz;
  print("td: ",f1);
  bar = foo(a);
  print("called foo");
  {
    vector[a] loc_td_b1 = bar;
    print("loc_td_b1",loc_td_b1);
  }
}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  vector[3] tpar_b = bar;
  print("tpar_b",tpar_b);
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  vector[3] gq_b1 = bar;
  row_vector[3] gq_c1 = baz;
  print("gq_b1",gq_b1);
  {
    vector[3] loc_gq_b1 = bar;
    row_vector[3] loc_gq_c1 = baz;
    print("loc_gq_b1",loc_gq_b1);
  }
}
