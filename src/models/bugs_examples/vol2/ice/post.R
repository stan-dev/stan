library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, -(1:3)]
poi <- as.mcmc(post)
summary(poi)

# The rest is for the ice example from JAGS

post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

Nage <- 13
K <- 11

## assuming the order of variables in samples.csv are the same as model
## specification file 
colnames(post) <- c(paste("alpha", 2:Nage, sep = ''), 
                    paste("beta", 1:K, sep = ''),
                    "sigma"); 

logRR <- matrix(0, ncol = K, nrow = nrow(post)) 
for (k in 1:K) {
   logRR[, k] <- (post[, paste("beta", k, sep = '')] - 
                  post[, paste("beta", 5, sep = '')]); 
} 

colnames(logRR) <- paste("logRR", 1:K, sep = ''); 

poi <- cbind(post[, "sigma"], logRR); 
colnames(poi)[1] <- "sigma"; 

poi <- as.mcmc(poi)
summary(poi) 


# copied from jags example 
"benchstats" <-
structure(c(0.0809343035, -1.210876809, -0.860664744, -0.510143228, 
-0.2013318802, 0, 0.140870989742000, 0.288454587000000, 0.443776121, 
0.5901469950, 0.782651164, 0.988174129, 0.04569394448987, 0.219802220574903, 
0.137193408720079, 0.0790044166801387, 0.0486954608384044, 0, 
0.0496285994768256, 0.0692647242226708, 0.0819167285040068, 0.106224387549485, 
0.148165497439102, 0.233518884916692, 0.0014449693986529, 0.00695075651779417, 
0.00433843651517856, 0.00249833901922237, 0.00153988567960890, 
0, 0.00156939411431010, 0.00219034290047075, 0.00259043440542299, 
0.00335911007712805, 0.00468540442559407, 0.00738451552999485, 
0.00157801515051251, 0.00807847963662094, 0.00511460556851936, 
0.00291727246116095, 0.00167524379199340, 0, 0.00160984056843626, 
0.00239048314799195, 0.00298720520765311, 0.00406722941370052, 
0.0059550763229483, 0.00900143318375036), .Dim = as.integer(c(12, 
4)), .Dimnames = list(c("sigma", "logRR[1]", "logRR[2]", "logRR[3]", 
"logRR[4]", "logRR[5]", "logRR[6]", "logRR[7]", "logRR[8]", "logRR[9]", 
"logRR[10]", "logRR[11]"), c("Mean", "SD", "Naive SE", "Time-series SE"
)))

print(benchstats) 
