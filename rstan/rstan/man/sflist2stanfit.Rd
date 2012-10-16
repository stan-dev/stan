
\name{sflist2stanfit} 
\alias{sflist2stanfit} 

\title{Merge a list of stanfit objects into one} 

\description{This function takes a list of \code{stanfit} objects and returns a
  consolidated \code{stanfit} object. The \code{stanfit} objects to be merged
  need to have the same configuration of iteration, warmup, and thin, besides
  being from the same model.  This could facilitate some parallel usage of RStan.
  For example, if we call \code{\link{stan}} by parallel and it returns a list of
  \code{stanfit} objects, this function can be used to create one \code{stanfit}
  object from the list. 
} 
 
\usage{
  sflist2stanfit(sflist) 
} 
 
\arguments{
  \item{sflist}{A list of \code{stanfit} objects.}
} 


\value{
  An S4 object of \code{stanfit} consolidated from all the input \code{stanfit}
  objects. 
}

\note{
  This function would do some check to see if the \code{stanfit} objects in the input list 
  can be merged. But the check is not sufficient. So generally, it is the
  user's responsibility to make sure the input is correct so that the merging
  makes sense. 

  The date in the new \code{stanfit} object is when it is merged. 

  \code{get_seed} function for the new consolidated \code{stanfit} object only returns
  the seed used in the first chain of the new object. 

  The sampler such as NUTS2 that is displayed in the printout by \code{print} 
  is the sampler used for the first chain. The \code{print} method assumes the samplers
  are the same for all chains. 

  The included \code{stanmodel} object, which includes the compiled model, 
  in the new \code{stanfit} object is from the first element of the input list. 

} 

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
} 

\seealso{
  \code{\link{stan}} 
} 

\examples{\dontrun{
library(rstan)
scode <- "
parameters {
  real y[2]; 
} 
model {
  y[1] ~ normal(0, 1);
  y[2] ~ double_exponential(0, 2);
} 
"
fit1 <- stan(model_code = scode, iter = 1000, verbose = FALSE, chains = 1) 
fit2 <- stan(model_code = scode, iter = 1000, verbose = FALSE, chains = 2) 
fit12 <- sflist2stanfit(list(a = fit1, b = fit2)) 

## parallel stan call 
library(parallel)

## this may not work on Windows 
sflist1 <- mclapply(1:4, mc.cores = 8, function(i) stan(fit = fit1, chains = 1))
fit3 <- sflist2stanfit(sflist1)
}} 
