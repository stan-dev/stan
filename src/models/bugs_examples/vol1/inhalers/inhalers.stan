# Inhaler: ordered categorical data 
## http://www.openbugs.info/Examples/Inhalers.html

## FIXME i(?): 
## Is there a way to specify say
## real a[3] with the restriction that a[1] < a[2] < a[3]? 
## 
## Now, it is specify as 
## a0 
## a0 + delta1 
## a0 + delta1 + delta2, with the restriction delta1 and delta > 0. 

## FIXE ii: 
## specify using categorical distribution directly 
## done (but x ~ categorical[p], in which x starts
## from 0). 


data {
  int(0,) N; 
  int(0,) T; 
  int(0,) G; 
  int(0,) Npattern; 
  int(0,) Ncum[16, 2]; 
  int(0,) pattern[16, 2]; 
  int(0,) Ncut;
  int treat[2, 2]; 
  int period[2, 2]; 
  int carry[2, 2]; 
} 

transformed data {
  # Construct individual response data from contingency table
  int group[N]; 
  int response[N, T]; 
  for (i in 1:Ncum[1, 1]) { 
    group[i] <- 1; 
    for (t in 1:T)  
      response[i, t] <- pattern[1, t]; 
  }
  for (i in (Ncum[1, 1] + 1):Ncum[1, 2]) { 
    group[i] <- 2; 
    for (t in 1:T)  
      response[i, t] <- pattern[1, t]; 
  }

  for (k in 2:Npattern) {
    for(i in (Ncum[k - 1, 2] + 1):Ncum[k, 1]) {
      group[i] <- 1; 
      for (t in 1:T)  
        response[i,t] <- pattern[k, t]; 
    }
    for(i in (Ncum[k, 1] + 1):Ncum[k, 2]) {
      group[i] <- 2; 
      for (t in 1:T)  
        response[i,t] <- pattern[k, t]; 
    }
  }
}

parameters {
  real(0,) sigmasq; 
  real beta; 
  real pi; 
  real kappa;
  real a0; 
  real(0,) delta1; 
  real(0,) delta2; 
  real b[N]; 
} 

transformed parameters {
  real a[Ncut];  // Ncut = 3
  real(0,) sigma; 
  a[1] <- a0; 
  a[2] <- a0 + delta1; 
  a[3] <- a0 + delta1 + delta2; 
  sigma <- sqrt(sigmasq); 
} 
model {
  real Q[N, T, Ncut]; 
  // real p[N, T, Ncut + 1]; 
  vector(Ncut + 1) p[N, T]; 
  real mu[G, T]; 

  for (g in 1:G) {
    for(t in 1:T) { 
      # logistic mean for group g in period t
      mu[g, t] <- beta * treat[g, t] * .5 + pi * period[g, t] * .5 + kappa * carry[g, t]; 
    }
  }                                                             

  for (i in 1:N) {
    for (t in 1:T) {
      for (j in 1:Ncut) {
        Q[i, t, j] <- inv_logit(-(a[j] + mu[group[i], t] + b[i])); 
      }

      p[i, t, 1] <- 1 - Q[i, t, 1];
      for (j in 2:Ncut)  
        p[i, t, j] <- Q[i, t, j - 1] - Q[i, t, j]; 
      p[i, t, (Ncut + 1)] <- Q[i, t, Ncut];
      
      response[i, t] - 1 ~ categorical(p[i, t]);
      // lp__ <- lp__ + log(p[i, t, response[i, t]]); 
    }
  }
  b ~ normal(0, sigma);

  beta ~ normal(0, 1000); 
  pi ~ normal(0, 1000); 
  kappa ~ normal(0, 1000); 

  a0 ~ normal(0, 1000);
  delta1 ~ normal(0, 1000); 
  delta2 ~ normal(0, 1000); 
 
  sigmasq ~ inv_gamma(0.001, 0.001);
}
generated quantities {
  real log_sigma;
  
  log_sigma <- log(sigma);
}