// The regularized horseshoe prior code from Appendix C.1 of Piironen and Vehtari (2017).
// Sparsity information and regularization in the horseshoe and other shrinkage priors.
// In Electronic Journal of Statistics, 11(2):5018-5051. https://projecteuclid.org/euclid.ejs/1513306866

data {
  int<lower=0> n;				      // number of observations
  int<lower=0> d;             // number of predictors
  int<lower=0,upper=1> y[n];	// outputs
  matrix[n,d] x;				      // inputs
  real<lower=0> scale_icept;	// prior std for the intercept
  real<lower=0> scale_global;	// scale for the half-t prior for tau
  real<lower=1> nu_global;	  // degrees of freedom for the half-t priors for tau
  real<lower=1> nu_local;		  // degrees of freedom for the half-t priors for lambdas
                              // (nu_local = 1 corresponds to the horseshoe)
  real<lower=0> slab_scale;   // for the regularized horseshoe
  real<lower=0> slab_df;
}

parameters {
  real beta0;
  vector[d] z;                // for non-centered parameterization
  real <lower=0> tau;         // global shrinkage parameter
  vector <lower=0>[d] lambda; // local shrinkage parameter
  real<lower=0> caux;
}

transformed parameters {
  vector[d] beta;                     // regression coefficients
  {
    vector[d] lambda_tilde;   // 'truncated' local shrinkage parameter
    real c = slab_scale * sqrt(caux); // slab scale
    lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2*square(lambda)));
    beta = z .* lambda_tilde*tau;
  }
}

model {
  // half-t priors for lambdas and tau, and inverse-gamma for c^2
  z ~ std_normal();
  lambda ~ student_t(nu_local, 0, 1);
  tau ~ student_t(nu_global, 0, scale_global*2);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  beta0 ~ normal(0, scale_icept);
  
  y ~ bernoulli_logit_glm(x, beta0, beta);
}