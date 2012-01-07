# Fire: Fire insurance claims: data distribution using lp__ directly 
#       since the composite lognormal-pareto distribution is not 
#       implemented. 
#  http://www.openbugs.info/Examples/Fire.html



// I need PI? 

// status: not work 
data {
  int(0,) N; 
  double x[N];
} 

parameters {
  double(0,) alpha;
  double(0,) sigma; 
  double(0,) theta;
} 

derived parameters {
  double mu;
  mu <- log(theta) - alpha * sigma * sigma; 
} 

model { 
  for (i in 1:N) {  
     double tmp; 
     double r; 
     tmp <- sqrt(2 * 3.141592653589) * alpha * sigma * Phi(alpha * sigma) * exp(0.5 * pow(alpha * sigma, 2));
     r <- tmp / (1 + tmp); 
     lp__ <- lp__ + step(theta - x[i]) * (log(r) - log(Phi(alpha * sigma)) + lognormal_log(x[i], mu, sigma));
     lp__ <- lp__ + step(x[i] - theta) * (log(1 - r) + pareto_log(x[i], theta, alpha)); 
  } 

  theta ~ gamma(.001, .001);   
  alpha ~ gamma(.001, .001); 
  sigma ~ gamma(.001, .001); 
}


  # xf <- xa * delta + xb * (1 - delta)
  # xa ~ dlnorm(mu, tau) I(, theta)
  # xb ~ dpar(alpha, theta)
  # delta ~ dbern(r)
  # tau <- 1 / pow(sigma, 2)
