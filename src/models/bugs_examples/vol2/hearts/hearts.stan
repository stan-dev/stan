# Hearts: a mixture model for count data
# http://www.openbugs.info/Examples/Hearts.html
# 

# integrate out the binary parameters in hearts.stan.0 

data {
  int<lower=0> N;
  int<lower=0> x[N];
  int<lower=0> y[N];
  int<lower=0> t[N];
} 

parameters {
  real alpha; 
  real delta; 
} 

transformed parameters {
  real<lower=0,upper=1> theta;
  theta <- inv_logit(delta); 
} 

model {
  real p; 
  p <- inv_logit(alpha); 

  alpha ~ normal(0, 100);
  delta ~ normal(0, 100); 
  for (i in 1:N) {
    # P(y_i = 0 | t_i) = theta + (1 - theta) * (1 - p)^{t_i}
    # p(y_i | t_i) = (1 - theta) * PMF_OF_BINOMIAL(t_i, y_i, p) for y_i = 1,2,...,t_i 

    if (y[i] == 0) 
     lp__ <- lp__ + log(theta + (1 - theta) * pow(1 - p, t[i]));  
    else 
     lp__ <- lp__ + log(1 - theta) + binomial_log(y[i], t[i], p);
  }
}

generated quantities {
  real beta;
  beta <- exp(alpha);
}
