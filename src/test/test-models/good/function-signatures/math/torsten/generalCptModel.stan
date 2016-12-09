functions {
  real[] ode(real t,
             real[] y,
             real[] theta,
             real[] x,
             int[] x_int) {
    real dydt[2];
    return dydt;
  }
}

data {
  int<lower = 1> nt;
  int nTheta;
  int<lower = 1> cmt[nt];
  int evid[nt];
  int addl[nt];
  int ss[nt];
  real amt[nt];
  real time[nt];
  real rate[nt];
  real ii[nt];

int<lower=1> T;
real y0_d[2];
real t0;
real ts[T];
real theta_d[1];
real x[0];
int x_int[0];
}

transformed data {
  int nCmt;
  real theta_data[nTheta, nt];
  matrix[nt, 2] x_data;

  nCmt = 2;

  x_data = generalCptModel_rk45(ode, nCmt, theta_data, time, amt, rate, ii, evid,
    cmt, addl, ss, 1e-8, 1e-8, 1e8);
  // x_data = generalCptModel_rk45(sho, nCmt, theta_data[0], time, amt, rate, ii, 
  //   evid, cmt, addl, ss, 1e-8, 1e-8, 1e8);
  
  x_data = generalCptModel_bdf(ode, nCmt, theta_data, time, amt, rate, ii, evid,
    cmt, addl, ss, 1e-8, 1e-8, 1e8);
}

parameters {
  real y_p;

real y0_p[2];
real theta_p[1];
}

transformed parameters {
  matrix[nt, nCmt] x_para;
  real theta_parm[nTheta, nt];
  matrix[nCmt, nCmt] K_parm[nt];

  x_para = generalCptModel_rk45(ode, nCmt, theta_parm, time, amt, rate, ii, evid,
    cmt, addl, ss, 1e-8, 1e-8, 1e8);
  // x_para = generalCptModel_rk45(ode, nCmt, theta_parm[0], time, amt, rate, ii, 
  //   evid, cmt, addl, ss, 1e-8, 1e-8, 1e8);
  
  x_para = generalCptModel_bdf(ode, nCmt, theta_parm, time, amt, rate, ii, evid,
    cmt, addl, ss, 1e-8, 1e-8, 1e8);
}

model {
	y_p ~ normal(0,1);
}
