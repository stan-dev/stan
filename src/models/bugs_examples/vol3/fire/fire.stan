/**
 * Fire: Fire insurance claims: data distribution incrementing log prob
 *       directly because the composite lognormal-pareto distribution
 *       is not built in.
 *
 *  http://www.openbugs.info/Examples/Fire.html
 *
 * see D. P. M. Scollnik (2007) 
 * http://www.tandfonline.com/doi/pdf/10.1080/03461230601110447
 *
 * the mixture used here is Equation (3.1) in the paper
 * with smooth density function so r in (3.1) is given 
 * in Equation (3.4) 
 */
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
  real tmp; 
  real log_tmp;
  real neg_log1p_temp;
  real log_Phi_alpha_times_sigma;

  theta ~ gamma(.001, .001);   
  alpha ~ gamma(.001, .001); 
  sigma ~ gamma(.001, .001); 

  tmp <- ( sqrt(2 * pi()) 
           * alpha 
           * sigma 
           * Phi(alpha * sigma) 
           * exp(0.5 * pow(alpha * sigma, 2)) );  
  log_tmp <- log(tmp);
  neg_log1p_temp <- - log1p(tmp);
  log_Phi_alpha_times_sigma <- log(Phi(alpha * sigma));
  for (i in 1:N) {  
    if (theta > x[i]) 
      lpa[i] <- ( log_tmp,
                  + neg_log1p_temp
                  - log_Phi_alpha_times_sigma
                  + lognormal_log(x[i], mu, sigma) ); 
    else 
      lpa[i] <- ( neg_log1p_temp
                  + pareto_log(x[i], theta, alpha) ); 
  } 
  increment_log_prob(sum(lpa));
}


