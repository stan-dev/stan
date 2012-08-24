
\name{rstan-package}
\alias{rstan-package}
\alias{rstan} 
\docType{package}
\title{RStan --- \R interface to Stan}
\description{
  \R interface to Stan, which is a C++ package for obtaining Bayesian
  inference using the No-U-turn sampler, a variant of Hamiltonian Monte
  Carlo (see \url{http:/stan.goglecode.com}). 
}

\details{
  \tabular{ll}{
    Package: \tab rstan\cr
    Type: \tab Package\cr
    Version: \tab 1.0\cr
    Date: \tab \Sexpr[eval=TRUE,results=rd,stage=build]{date()}\cr
    License: \tab GPL-3 \cr
  }

  The RStan package provides an interface to Stan. 
  See the manual of Stan for more information. 

}
\author{
  \tabular{ll}{
  Author:\tab Jiqiang Guo <guojq28@gmail.com>   \cr 
         \tab Marcus Brubaker <mbrubake@cs.toronto.edu>   \cr 
         \tab Bob Carpenter <carp@alias-i.com>   \cr 
         \tab Ben Goodrich <goodrich.ben@gmail.com>  \cr 
         \tab Matt Hoffman <mdhoffma@cs.princeton.edu>  \cr 
         \tab Daniel Lee <bearlee@alum.mit.edu>  \cr 
         \tab Michale Malecki <malecki@gmail.com>  \cr 
         \tab Andrew Gelman <gelman@stat.columbia.edu> \cr 
  } 

  \tabular{ll}{
  Maintainer:\tab Jiqiang Guo <guojq28@gmail.com>    \cr
  } 
}
\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://mc-stan.org}. 
}

\examples{\dontrun{ 
library(rstan) 
cat("Stan version: ", stanc.version(), sep = ''); 

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
fit <- stan(model.code = stanmodelcode, model.name = "example", 
            data = dat, n.iter = 2012, n.chains = 3, sample.file = 'norm.csv',
            verbose = TRUE) 
print(fit)
}}  
\keyword{rstan} 
\keyword{package} 

