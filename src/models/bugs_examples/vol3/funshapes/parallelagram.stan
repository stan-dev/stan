# http://www.openbugs.info/Examples/Funshapes.html

# status: working 

parameters {
  real[0, 1] x; 
  real[-1, 1] y; 
} 

model {
  // lp__ <- log(step(1 - x * x - y * y)); 
  lp__ <- lp__ + log(fmax(0, x + y)); 
  lp__ <- lp__ + log(fmax(0, 1 - x - y)); 
} 
