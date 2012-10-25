
\name{as.array}
\alias{is.array.stanfit}
\alias{as.array.stanfit}
\alias{as.matrix.stanfit}
\alias{dim.stanfit}
\alias{dimnames.stanfit}

\title{Create array or matrix objects from samples in \code{stanfit} object} 


\description{The samples without warmup included in a \code{\linkS4class{stanfit}} 
  object can be coerced to \code{array} or \code{matrix}} 

\usage{
  \method{as.array}{stanfit}(x, \dots) 
  \method{as.matrix}{stanfit}(x, \dots) 
  \method{is.array}{stanfit}(x) 
  \method{dim}{stanfit}(x)
  \method{dimnames}{stanfit}(x)
} 

\arguments{
  \item{x}{An object of S4 class \code{stanfit}} 
  \item{\dots}{Additional parameters that can be passed to \code{extract} 
    for extracting samples from \code{stanfit} object. Now \code{pars}
    is the only additional parameter.  
  } 
} 

\details{
  \code{as.array} and \code{as.matrix} can be applied to a \code{stanfit}
  object to coerce the samples without warmup to \code{array} or \code{matrix}. 
  The array has three dimensions: iteration, chain id, parameter. 
  In the case of matrix, all chains for the same parameter are concatenated. 

  \code{dim} and \code{dimnames} return the dim and dimnames of the 
  array object to be created. 

  \code{is.array} returns \code{TRUE} for \code{stanfit} objects that includes 
  samples; otherwise \code{FALSE}. 

  When the \code{stanfit} object does not containing samples, empty objects
  are returned from \code{as.array}, \code{as.matrix}, \code{dim}, and 
  \code{dimnames}. 
} 
  
\seealso{
  S4 class \code{\linkS4class{stanfit}} and its method \code{\link{extract}} 
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

dim(fit)
dimnames(fit)
is.array(fit) 
a <- as.array(fit)
m <- as.matrix(fit)
}}  
