## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

NT <- 17

dL0 <- post[, 1 + (1:NT)]; 
beta <- post[, 1]; 

S_treat <- matrix(0, ncol = NT, nrow = nrow(post))
S_placebo <- matrix(0, ncol = NT, nrow = nrow(post))


pow <- function(x, y) x^y; 

# the Survior function for in iteration 
Sfun <- function(p) { 
    beta <- p[1]; 
    dL0 <- p[-1]; 
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

