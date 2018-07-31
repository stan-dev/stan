functions{
  real twoCptModelODE(real t,
                      real[] x,
                      real[] parms,
                      real[] rdata,
                      int[] idata){
    real dxdt[2];
    return dxdt[2];
  }
}
model {
  real x[2, 2]
    = integrate_ode_adams(twoCptModelODE,
                        {1, 1.3}, 1.0, { 2.2, 3 }, { 1.0 }, { 1.0 }, { 2 },
                        10, 10, 10);
}
