# FIRST RUN SIMULATION to get a value to fit:
# source("gp-sim.R");  # first run simulation 

fit_predict <- stan(file="gp-predict.stan",   
                    data=list(N1=N1,x1=x1,y1=y1,N2=N2,x2=x2),
                    iter=200, chains=3);
fit_predict_ss <- extract(fit_predict, permuted=TRUE);
print(fit_predict);

# plot fits vs. simulated value from which fits drawn
df <- data.frame(x2=x2, y2_samp=fit_predict_ss$y2[100,]);
plot <- qplot(x2,y2_samp, data=df, xlim=c(-8,8), ylim=c(-4,4));


