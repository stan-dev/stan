transformed data {
  real udummy;
  { 
    real v;
    real vv[10];
    vector[12] vvv;
    matrix[10,10] vvvv;
    matrix[3,4] vvvvv[5];
    print("transformed data: ",v,"  ",vv[1],"  ",vvv[2],"  ",vvvv[3,4],"  ",vvvvv[1,2,3]);
  }
}
parameters {
  real y;
}
transformed parameters {
  real wdummy;
  { 
    real w;
    real ww[10];
    vector[12] www;
    matrix[10,10] wwww;
    matrix[3,4] wwwww[5];
    print("transformed parameters: ",w," ",ww[1]," ",www[2]," ",wwww[3,4]," ",wwwww[1,2,3]);
  }
}
model {
  real z;
  real zz[10];
  vector[12] zzz;
  matrix[10,10] zzzz;
  matrix[10,10] zzzzz[10];
  print("model: ",z," ",zz[1]," ",zzz[2]," ",zzzz[3,4]," ",zzzzz[3,4,5]); 
  y ~ normal(0,1);
}
