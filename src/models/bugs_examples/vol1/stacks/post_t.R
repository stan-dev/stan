library(coda)
source('stacks.Rdata')
z <- apply(x, 2, FUN = function(x) { (x - mean(x)) / sd(x); })
post <- read.csv(file = "samples_t.csv", header = TRUE, comment.char = '#')

# the degree of freedom specified for the student_t 
# dsn in the model 
dof <- 4

sigma <- sqrt(post[, "sigmasq"]) * dof / (dof - 2)

mystep <- function(i) ifelse(i >= 0, 1, 0)

mu <- matrix(0, ncol = N, nrow = nrow(post))
outlier <- mu
for (i in 1:nrow(post)) { 
    mu[i, ] <- (post[i, "beta0"] + 
                post[i, "beta.1"] * z[, 1] + 
                post[i, "beta.2"] * z[, 2] + 
                post[i, "beta.3"] * z[, 3]) 
    tmp <- (Y - mu[i,]) / sigma[i]
    outlier[i, ] <- mystep(tmp - 2.5) + mystep(-(tmp + 2.5))
} 

b1 <- post[, "beta.1"] / sd(x[, 1])
b2 <- post[, "beta.2"] / sd(x[, 2])
b3 <- post[, "beta.3"] / sd(x[, 3])
b0 <- post[, "beta0"] - b1 * mean(x[, 1]) - b2 * mean(x[, 2]) - b3 * mean(x[, 3])

poi <- cbind(b1, b2, b3, b0, outlier[, c(3, 4, 21)], sigma)
colnames(poi) <- c('b0', 'b1', 'b2', 'b3', paste('outlier', c(3, 4, 21), sep = ''), "sigma") 
summary(as.mcmc(poi))
