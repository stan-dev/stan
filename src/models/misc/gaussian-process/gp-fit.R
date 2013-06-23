
# FIRST RUN SIMULATION to get a value to fit:
# source("gp-sim.R");  # first run simulation 

y_samp <- fit_sim_ss$y[100,]

fit_fit <- stan(file="gp-fit.stan", data=list(x=x,N=N,y=y_samp),
                 iter=200, chains=3);

fit_fit_ss <- extract(fit_fit, permuted=TRUE);

traceplot(fit_fit);

# histograms of fits nice here
