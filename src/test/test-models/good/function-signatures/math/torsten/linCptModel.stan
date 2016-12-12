data {
  int<lower = 1> nt;
  int nTheta;
  int nCmt;
  int<lower = 1> cmt[nt];
  int evid[nt];
  int addl[nt];
  int ss[nt];
  real amt[nt];
  real time[nt];
  real rate[nt];
  real ii[nt];
}

transformed data {
  real theta_data[nTheta, nt];
  matrix[nCmt, nCmt] K_data[nt];
  matrix[nt, nCmt] x_data;

  x_data = linCptModel(K_data, theta_data, time, amt, rate, ii, evid, cmt, addl, ss);
  x_data = linCptModel(K_data[0], theta_data[0], time, amt, rate, ii, evid, cmt, addl, ss);
}

parameters {
  real y_p;
}

transformed parameters {
  matrix[nt, nCmt] x_para;
  real theta_parm[11, nt];
  matrix[nCmt, nCmt] K_parm[nt];

  x_para = linCptModel(K_parm, theta_parm, time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = linCptModel(K_parm, theta_parm[0], time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = linCptModel(K_parm[0], theta_data, time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = linCptModel(K_parm[0], theta_data[0], time, amt, rate, ii, evid, cmt, addl, ss);
}

model {
	y_p ~ normal(0,1);
}
