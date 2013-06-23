library("rstan");
source("koyck.data.R");
fit <- stan(file="koyck.stan", data=c("T","y","x"), iter=500, chains=4);
