##  simulate samples from student t distribution 
## 
# http://www.openbugs.info/Examples/t-df.html

transformed data {
  int d; 
  d <- 4; 
} 

parameters {
  real y[1000]; 
} 

model {
  for (i in 1:1000) {
    y[i] ~ student_t(d, 0, 1); 
  } 
} 
