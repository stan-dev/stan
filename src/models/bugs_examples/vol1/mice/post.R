library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

M <- 4  

irr_control <- post[, "beta.1"]; 
veh_control <- post[, "beta.2"] - post[, "beta.1"]; 
test_sub <- post[, "beta.3"] - post[, "beta.1"]; 
post_control <- post[, "beta.4"] - post[, "beta.1"]; 

medians <- matrix(0, ncol = M, nrow = nrow(post)) 

for (j in 1:M) {
    betaj <- paste("beta.", j, sep = '')  
    medians[, j] <- (log(2) * exp(-post[, betaj])) ^ (1 / post[, "r"])   
    names(medians[, j]) <- paste("median", j, sep = '')  
} 

poi <- cbind(irr_control, veh_control, test_sub, post_control, post[, "r"], medians) 
colnames(poi)[1:5] <- c("irr_control", "veh_control", "test_sub", "post_control", "r");
colnames(poi)[6:(5 + M)] <- paste("median", 1:M, sep = '')
poi <- as.mcmc(poi[, -1])
summary(poi) 

# copied from jags example 
"benchstats" <-
structure(c(-1.19050455833999, -0.358970521867199, 0.398739260972799, 
3.282505832, 24.4461747099999, 35.2348815999999, 27.29264683, 
21.6708317800001, 0.368704952159327, 0.349321013036338, 0.345746877312227, 
0.328831602502753, 1.83527934356389, 3.1409202416556, 2.24902141666842, 
1.69732778227839, 0.00368704952159327, 0.00349321013036338, 0.00345746877312227, 
0.00328831602502753, 0.0183527934356389, 0.031409202416556, 0.0224902141666842, 
0.0169732778227839, 0.00370973573935992, 0.00345612641578039, 
0.0035520553624397, 0.00593038770326027, 0.0196213658552674, 
0.0305511869544568, 0.0234729427497584, 0.017304934105965), .Dim = as.integer(c(8, 
4)), .Dimnames = list(c("veh.control", "test.sub", "pos.control", 
"r", "median[1]", "median[2]", "median[3]", "median[4]"), c("Mean", 
"SD", "Naive SE", "Time-series SE")))

print(benchstats) 
