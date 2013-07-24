# Fire: Fire insurance claims: data distribution using lp__ directly 
#       since the composite lognormal-pareto distribution is not 
#       implemented. 
#  http://www.openbugs.info/Examples/Fire.html

## see D. P. M. Scollnik (2007) 
## http://www.tandfonline.com/doi/pdf/10.1080/03461230601110447

## the mixture used here is Equation (3.1) in the paper
## with smooth density function so r in (3.1) is given 
## in Equation (3.4) 

data {
  int<lower=0> N; 
  real x[N];
} 

parameters {
  real<lower=0> alpha;
  real<lower=0> sigma; 
  real<lower=0> theta;
} 

transformed parameters {
  real mu;
  mu <- log(theta) - alpha * sigma * sigma; 
} 

model { 
  real lpa[N];
  for (i in 1:N) {  
    real tmp; 
    tmp <- sqrt(2 * pi()) * alpha * sigma * Phi(alpha * sigma) * exp(0.5 * pow(alpha * sigma, 2));
    if (theta > x[i]) 
      lpa[i] <- log(tmp) - log1p(tmp) - log(Phi(alpha * sigma)) + lognormal_log(x[i], mu, sigma); 
    else 
      lpa[i] <- -log1p(tmp) + pareto_log(x[i], theta, alpha); 
  } 
  lp__ <- lp__ + sum(lpa);

  theta ~ gamma(.001, .001);   
  alpha ~ gamma(.001, .001); 
  sigma ~ gamma(.001, .001); 
}


  # xf <- xa * delta + xb * (1 - delta)
  # xa ~ dlnorm(mu, tau) I(, theta)
  # xb ~ dpar(alpha, theta)
  # delta ~ dbern(r)
  # tau <- 1 / pow(sigma, 2)
