data {
  int<lower = 1> nt;
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
  real theta_data[7, nt];
  matrix[nt, 2] x_data;

  x_data = PKModelOneCpt(theta_data, time, amt, rate, ii, evid, cmt, addl, ss);
  x_data = PKModelOneCpt(theta_data[0], time, amt, rate, ii, evid, cmt, addl, ss);
}

parameters {
  real y_p;
}

transformed parameters {
  matrix[nt, 2] x_para;
  real theta_parm[7, nt];

  x_para = PKModelOneCpt(theta_parm, time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = PKModelOneCpt(theta_parm[0], time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = PKModelOneCpt(theta_data, time, amt, rate, ii, evid, cmt, addl, ss);
  x_para = PKModelOneCpt(theta_data[0], time, amt, rate, ii, evid, cmt, addl, ss);
}

model {
	y_p ~ normal(0,1);
}
