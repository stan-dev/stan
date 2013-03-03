mcode <- '
  transformed data {
    vector[5] v; 
    v[1] <- 1;
    v[2] <- 2;
    v[3] <- 3;
    v[4] <- 4;
    v[5] <- 5;
  } 
  parameters {
    simplex[5] y;
  } 
  model {
    y ~ dirichlet(v);
  }
'

library(rstan)
sm <- stan_model(model_code = mcode)
# fit <- sampling(sm, chains = 3, seed = 12345)
mod <- sm@dso@.CXXDSOMISC$module 

model_cppname <- paste0("stan_fit4", sm@model_cpp$model_cppname)
stan_fit_cpp_module <- eval(call("$", mod, model_cppname))
sampler <- new(stan_fit_cpp_module, list(), sm@dso@.CXXDSOMISC$cxxfun)
fit <- sampling(sm, data = list())

# from constrained space to the unconstrained 
up <- sampler$unconstrain_pars(list(y = c(0.1, 0.2, 0.3, 0.3, 0.1)))

p <- sampler$constrain_pars(up) 
n <- sampler$num_pars_unconstrained()
cat("n=", n, "\n")
print(p)

K <- 5
fit <- stan(model_code = mcode)
lp <- log_prob(fit, rep(0, K - 1))
gr <- grad_log_prob(fit, rep(0, K- 1))

nupar <- get_num_upars(fit)
cat("nupar=", nupar, "\n")
up2 <- unconstrain_pars(fit, list(y = rep(1 / K, K))) 
# up2 should be rep(0, K - 1) 
p2 <- constrain_pars(fit, up2)

if (all.equal(unname(unlist(p2)), rep(0.2, 5))) {
  cat('found no problem with transforming parameters\n')
} 

### do a simple optimization problem 
opcode <- '
parameters {
  real y;
}
model {
  lp__ <- log(square(y - 5) + 1);
}
'

# specifying iter = 0 intendedly
opfit <- stan(model_code = opcode, chains = 1, iter = 0)


tfun <- function(y) log_prob(opfit, y)
tgrfun <- function(y) grad_log_prob(opfit, y)
or <- optim(1, tfun, tgrfun, method = 'BFGS')
print(or)
