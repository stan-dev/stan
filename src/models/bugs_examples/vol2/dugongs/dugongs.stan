// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 2: Dugongs 
// 
//
// note that in the original example, the parameter 
// called gamma is called lambda here.  
// 
// If we specify lambda as ``double lambda'' instead of
// ``double(.5, 1) lambda'', it looks like stan would 
// go to an infinite loop.  (Fri Nov 25 15:58:21 EST 2011)
//


data {
   int(0,) N; 
   double x[N]; 
   double Y[N]; 
} 
parameters {
   double alpha; 
   double beta;  
   double(.5, 1) lambda; // orginal gamma in the JAGS example  
   // double lambda; 
   double(0,) sigma; 
} 
derived parameters {
   double tau; 
   double U3; 
   tau <- 1 / (sigma * sigma);
   U3 <- logit(lambda);
} 
model {
   for (i in 1:N) 
      Y[i] ~ normal(alpha - beta * pow(lambda, x[i]), sigma);
    
   alpha ~ normal(0.0, 1000); 
   beta ~ normal(0.0, 1000); 
   lambda ~ uniform(.5, 1); 
   tau ~ gamma(.0001, .0001); 
   // sigma ~ gamma(2, .001); // a different prior 
}

