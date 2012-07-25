## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 
NT <- 17
beta <- post[, "beta"]  

S_treat <- matrix(0, ncol = NT, nrow = nrow(post))
S_placebo <- matrix(0, ncol = NT, nrow = nrow(post))


pow <- function(x, y) x^y; 

dL0_idx <- grep("dL0.[:digits:]*", colnames(post))

# the Survior function for in iteration 
Sfun <- function(p) { 
    beta <- p["beta"]; 
    dL0 <- p[dL0_idx]; 
    NT <- length(dL0) 
    S.treat <- numeric(NT)  
    S.placebo <- numeric(NT)  
    for (j in 1:length(dL0)) { 
        S.treat[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * -0.5));
        S.placebo[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * 0.5));	
    } 
    c(S.placebo, S.treat); 
} 

S <- apply(post, 1, Sfun)

poi <- cbind(t(S), beta);
colnames(poi) <- c(paste("S.placebo", 1:NT, sep = ''), 
                   paste("S.treat", 1:NT, sep = ''), "beta")

summary(as.mcmc(poi)) 

