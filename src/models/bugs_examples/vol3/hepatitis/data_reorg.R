source('hepatitis.data.R') 
# Yvec <- c(Y[, 1], Y[, 2], Y[, 3]) 
Yvec <- as.vector(Y) 
# tvec <- c(t[, 1], t[, 2], t[, 3]) 
tvec <- as.vector(t) 
idxn <- rep(1:N, T); 


naidx <- which(is.na(Yvec)) 

Yvec1 <- Yvec[-naidx]; 
idxn1 <- idxn[-naidx]; 
tvec1 <- tvec[-naidx]; 

N1 <- length(Yvec1) 


dump(c("N", "Yvec1", "idxn1", "tvec1", "N1", "y0"), file = 'hepatitis2.data.R');
