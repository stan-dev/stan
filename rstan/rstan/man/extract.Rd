\name{extract-methods}
\docType{methods}
\alias{extract} 
\alias{extract-methods}
\alias{extract,stanfit-method}
%\alias{extract,stanfit,character-method}

\title{\code{extract}: extract samples from fitted Stan model}
\description{
  Extract samples from a fitted model represented by an 
  instance of class \code{stanfit}.
}

\usage{
  %% extract(object, \dots)  
  \S4method{extract}{stanfit}(object, pars, permuted = TRUE, inc_warmup = FALSE) 
} 

\section{Methods}{
  \describe{
    \item{extract}{\code{signature(object = "stanfit")}}{ Extract samples for a
      fitted model defined by class \code{stanfit}.  
    }
  }
}

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
  \item{pars}{An object of class \code{"character"} providing the parameter
    names (or other quantity names) of interest. If not specified,
    all parameters and other quantities are used. The log-posterior with
    name \code{lp__} is also included in the samples. 
  } 
  \item{permuted}{\code{TRUE} of \code{FALSE}. If \code{TRUE}, draws
    after the \emph{warmup} period in each chain are \emph{permuted} and \emph{merged}. 
    If \code{FALSE}, the original order are kept. For each \code{stanfit}
    object, the permutation is fixed (that is, extracing samples a second
    time will give the same sequence of iterations as the previous).}
  \item{inc_warmup}{\code{TRUE} of \code{FALSE}. The argument matters
    only if \code{permuted} is \code{FALSE}. If \code{TRUE}, warmup
    draws are included; otherwise excluded.}
}
\value{
  When \code{permuted = TRUE}, this function returns a named list,  
  every element of which is an array representing samples for a parameter
  merged from all chains. 
  When \code{permuted = FALSE}, an array is returned; the first
  dimension is for the iterations; the second for the number of chains; the 
  third for the parameters. Vectors and arrays are expanded to one 
  parameter per cell, with names indicating the third dimension.
  See examples below and comments in the code. 
}

\examples{\dontrun{
ex_model_code <- '
  parameters {
    real alpha[2,3];
    real beta[2]; 
  } 
  model {
    for (i in 1:2) for (j in 1:3) 
      alpha[i, j] ~ normal(0, 1); 
    for (i in 1:2) 
      beta ~ normal(0, 2); 
  } 
'

## fit the model 
fit <- stan(model_code = ex_model_code, chains = 4) 

## extract alpha and beta with 'permuted = TRUE' 
fit_ss <- extract(fit, permuted = TRUE) # fit_ss is a list 
## list fit_ss should have elements with name 'alpha', 'beta', 'lp__'
alpha <- fit_ss$alpha  
beta <- fit_ss$beta 
## or extract alpha by just specifying pars = 'alpha' 
alpha2 <- extract(fit, pars = 'alpha', permuted = TRUE)$alpha 

print(identical(alpha, alpha2)) 

## get the samples for alpha[1,1] and beta[2] 
alpha_11 <- alpha[, 1, 1] 
beta_2 <- beta[, 2] 

## extract samples with 'permuted = FALSE' 
fit_ss2 <- extract(fit, permuted = FALSE) # fit_ss2 is an array  

## the dimensions of fit_ss2 should be  
## "# of iterations * # of chains * # of parameters"
dim(fit_ss2) 

## since the third dimension of `fit_ss2` indicates 
## parameters, the names should be 
##  alpha[1,1], alpha[2,1], alpha[1,2], alpha[2,2], 
##  alpha[1,3], alpha[2,3], beta[1], beta[2], and lp__ 
## `lp__` (the log-posterior) is always included 
## in the samples.  
dimnames(fit_ss2) 
}} 

\seealso{
  S4 class \code{\linkS4class{stanfit}} and \link{as.array.stanfit}
} 

\keyword{methods}
