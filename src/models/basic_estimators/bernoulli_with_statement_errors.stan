data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
}

transformed data {
	//err 1
}

parameters {
  real<lower=0,upper=1> theta;
} 

transformed parameters {
	err 2
}

model {
  theta ~ beta(1,1);
  for (n in 1:N) {
    y[n] ~ bernoulli(theta);  
  }
}

generated quantities {
	
}
