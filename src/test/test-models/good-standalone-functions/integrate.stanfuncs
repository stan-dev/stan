functions {

  vector integrand(vector x) {
    return exp(-square(x));
  }

  real[] integrand_ode(real r, real[] f, real[] theta, real[] x_r, int[] x_i) {
    real df_dx[1];
    real x = logit(r);
    df_dx[1] = exp(-square(x)) * 1/(r * (1-r));
    return(df_dx);
  }

  real ode_integrate() {
    int x_i[0];
    // ok:
    //return(integrate_ode_rk45(integrand_ode, rep_array(0.0, 1),
    //1E-5, rep_array(1.0-1E-5, 1), rep_array(0.0, 0), rep_array(0.0,
    //0), x_i)[1,1]);
    // not ok
    return(integrate_ode_bdf(integrand_ode, rep_array(0.0, 1), 1E-5, rep_array(1.0-1E-5, 1), rep_array(0.0, 0), rep_array(0.0, 0), x_i)[1,1]);
  }

}
data {
}
model {}