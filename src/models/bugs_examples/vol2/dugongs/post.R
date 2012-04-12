## take a look at the samples and compare with results computed 
## in other program. 


library(coda) 
logit <- function(x) log(x / (1 - x)); 

post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

U3 <- logit(post[, 'lambda']) 
names(U3) <- "U3"; 

sigma <- 1 / sqrt(post[, "tau"])
names(sigma) <- "sigma" 

poi <- cbind(U3, post[, c("alpha", "beta", "lambda")], sigma); 

poi <- as.mcmc(poi)
summary(poi) 


# copied from jags example 
# changed the name of gamma to lambda 
"benchstats" <-
structure(c(1.86734593050999, 2.65550706999999, 0.976866162000004, 
0.862622853, 0.0993965308999997, 0.288028864815654, 0.0738703034051412, 
0.0788082243785414, 0.0361326586182234, 0.0155265477362125, 0.00288028864815654, 
0.000738703034051412, 0.000788082243785414, 0.000361326586182234, 
0.000155265477362125, 0.00505762524854399, 0.00129773683846219, 
0.00135515901718672, 0.000634560717168502, 0.000184674606469514
), .Dim = as.integer(c(5, 4)), .Dimnames = list(c("U3", "alpha", 
"beta", "lambda", "sigma"), c("Mean", "SD", "Naive SE", "Time-series SE"
)))

print(benchstats) 
