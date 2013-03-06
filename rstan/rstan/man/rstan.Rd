
\name{rstan-package}
\alias{rstan-package}
\alias{rstan} 
\docType{package}
\title{RStan --- \R interface to Stan}
\description{
  \R interface to Stan, which is a C++ package for obtaining Bayesian
  inference using the No-U-turn sampler, a variant of Hamiltonian Monte
  Carlo (see \url{http://mc-stan.org/}).
}

\details{
  \tabular{ll}{
    Package: \tab rstan\cr
    Type: \tab Package\cr
    Version: \tab 1.2.0\cr
    Date: \tab \Sexpr[eval=TRUE,results=rd,stage=build]{date()}\cr
    License: \tab GPL-3 \cr
  }

  The RStan package provides an interface to Stan. 
  For more information on Stan and its modeling language,
  see the \emph{Stan Modeling Language User's Guide and Reference Manual},   
  which is available \url{http://mc-stan.org/}.

}
\author{
  \tabular{ll}{
  Author:\tab Jiqiang Guo <guojq28@gmail.com>   \cr 
         \tab Michael Betancourt <betanalpha@gmail.com> \cr 
         \tab Marcus Brubaker <mbrubake@cs.toronto.edu>   \cr 
         \tab Bob Carpenter <carp@alias-i.com>   \cr 
         \tab Ben Goodrich <bg2382@columbia.edu> \cr 
         \tab Matt Hoffman <mdhoffma@cs.princeton.edu>  \cr 
         \tab Daniel Lee <bearlee@alum.mit.edu>  \cr 
         \tab Michael Malecki <malecki@gmail.com>  \cr 
         \tab Andrew Gelman <gelman@stat.columbia.edu> \cr 
  } 

  \tabular{ll}{
  Maintainer:\tab Jiqiang Guo <guojq28@gmail.com>    \cr
  } 
}
\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
}

\examples{\dontrun{ 
library(rstan) 
cat("Stan version:", stan_version(), "\n")

stanmodelcode <- "
data {
  int<lower=0> N;
  real y[N];
} 

parameters {
  real mu;
} 

model {
  mu ~ normal(0, 10);
  y ~ normal(mu, 1); 
} 
"

y <- rnorm(20) 
dat <- list(N = 20, y = y); 
fit <- stan(model_code = stanmodelcode, model_name = "example", 
            data = dat, iter = 2012, chains = 3, sample_file = 'norm.csv',
            verbose = TRUE) 
print(fit)

# extract samples 
e <- extract(fit, permuted = TRUE) # return a list of arrays 
mu <- e$mu 

m <- extract(fit, permuted = FALSE, inc_warmup = FALSE) # return an array 
print(dimnames(m))

# using as.array directly on stanfit objects 
m2 <- as.array(fit)

}}  

\seealso{
  \code{\link{stan}}, \code{\linkS4class{stanfit}} 
} 
\keyword{rstan} 
\keyword{package} 

