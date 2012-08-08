# http://www.openbugs.info/Examples/Funshapes.html

parameters {
  real[-1, 1] x; 
  real[-1, 1] y; 
} 

model {
  // lp__ <- log(step(1 - x * x - y * y)); 
  lp__ <- log(fmax(0, 1 - x * x - y * y)); 
  
} 
