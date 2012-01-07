# http://www.openbugs.info/Examples/Funshapes.html
# square minus circle 

## status: not working (stan is hanging over there)

parameters {
  double(-1, 1) x; 
  double(-1, 1) y; 
} 

model {
  lp__ <- log(fmax(x * x + y * y - 1, 0)); 
} 
