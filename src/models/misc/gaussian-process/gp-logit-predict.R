# FIRST RUN SIMULATION to get a value to fit:
# source("gp-sim.R");  # first run simulation 

fit_logit_predict <- stan(file="gp-logit-predict.stan",   
                          data=list(N1=N1,x1=x1,z1=z1,N2=N2,x2=x2),
                          iter=200, chains=3);
fit_logit_predict_ss <- extract(fit_logit_predict, permuted=TRUE);

print(fit_predict);


df <- data.frame(x2 = x2, y2_samp = fit_logit_predict_ss$y2[93,]);
plot <- qplot(x2, y2_samp, data = df, xlim=c(-8,8), ylim=c(-4,4));
