/**
 * Minimal example derived from jimsinmss example brought
 * up on stan-users mailing list, Feb 2015.
 */
functions {
  real[] foo(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    return rep_array(1.0,1);
  }
}
transformed data {
  real y;
  real t[2,2];
  y <- integrate_ode(foo,
                     rep_array(1.0,1),
                     1.0,
                     t[1],          
                     rep_array(1.0,1),
                     rep_array(1.0,1),
                     rep_array(1,1))[1,1];
}
model {
}
