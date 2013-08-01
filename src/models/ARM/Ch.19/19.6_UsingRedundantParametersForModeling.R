## Read, clean the pilots data, redefine variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/schools

library(rstan)
schools <- read.table ("schools.dat", header=TRUE)
attach(schools)

N <- 8
y <- c(28,8,-3,7,-1,1,18,12)
sigma.y <- c(15,10,16,11,9,11,10,18)

if (!exists("schools.sm")) {
    if (file.exists("schools.sm.RData")) {
        load("schools.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("schools.stan", model_name = "schools")
        schools.sm <- stan_model(stanc_ret = rt)
        save(schools.sm, file = "schools.sm.RData")
    }
}
dataList.1 <- list(N=N, sigma_y=sigma.y,y=y)
schools.sf1 <- sampling(schools.sm, dataList.1)
print(schools.sf1, pars = c("theta", "lp__"))
