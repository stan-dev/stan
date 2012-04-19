
# read the simulated student t samples 
library(coda)
poinames <- c("alpha", "sigma", "theta", "mu") 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, poinames] 

summary(as.mcmc(post)) 


## from http://www.openbugs.info/Examples/Fire.html
# mean   sd   MC_error   val2.5pc   median   val97.5pc   start   sample
# alpha   1.328   0.03179   7.257E-4   1.267   1.328   1.392   1001   10000
# sigma   0.1971   0.01209   5.755E-4   0.1745   0.1966   0.2219   1001   10000
# theta   1.209   0.03123   0.001505   1.153   1.207   1.273   1001   10000
