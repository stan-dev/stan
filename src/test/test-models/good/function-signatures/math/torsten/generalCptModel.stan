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
  int nCmt = 2;
  real theta_data[nt, nTheta];
  real biovar_data[nt, nCmt];
  real tlag_data[nt, nCmt];
  matrix[nt, nCmt] x_data;

  nCmt = 2;

  // RK45
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);   

  // BDF    
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_data = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);


}

parameters {
  real y_p;

  real y0_p[2];
  real theta_p[1];
}

transformed parameters {
  real theta_parm[nt, nTheta];
  real biovar_parm[nt, nCmt];
  real tlag_parm[nt, nCmt];
  matrix[nt, nCmt] x_parm;

  // RK45
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_rk45(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);


  // BDF
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data[0], biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_data, biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_parm, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_parm[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_parm[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_data, tlag_data[0], 1e-8, 1e-8, 1e8);

  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm[0], biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_data, 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm[0], tlag_data[0], 1e-8, 1e-8, 1e8);
  x_parm = generalOdeModel_bdf(ode, nCmt, time, amt, rate, ii, evid, cmt,
    addl, ss, theta_parm, biovar_parm, tlag_data[0], 1e-8, 1e-8, 1e8);
}

model {
	y_p ~ normal(0,1);
}
