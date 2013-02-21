\name{monitor}
\alias{monitor}
\title{Compute the summary for MCMC simulation samples and monitor the convergence} 
\description{For a 3-d array (the number of iterations * the number of chains *
  the number of parameters) of MCMC simulation samples, this function computes 
  the summaries such as mean, standard deviation, standard error of the mean,
  and quantiles.  And for monitoring the convergence, split Rhat and the
  effective sample size are also computed.}

\usage{
monitor(sims, warmup = floor(dim(sims)[1]/2), 
          probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
          digits_summary = 1, print = TRUE, ...)
}
\arguments{
  \item{sims}{A 3-dimension array of samples simulated from any MCMC algorithm. 
    The first dimension is for the number of iterations; the second for the 
    number of chains; the third for the parameters.}
  \item{warmup}{The number of warmup iterations that would be excluded 
    for computing the summaries; default to half of the total number 
    of iterations.}
  \item{probs}{Quantiles of interest; defaults to \code{"c(0.025,0.25,0.5,0.75,0.975)"}}
  \item{digits_summary}{The number of significant digits for printing out
    the summary; defaults to 1. The effective sample size is always rounded to
    integers.} 
  \item{\dots}{Additional arguments for the underlying \code{print} method.} 
  \item{print}{Logical; indicating whether to print the summary.} 
}
\details{
  Similar to the \code{print} function for \code{stanfit} object, 
  the function prints out a summary for the simulated samples. 
  In particular, for monitoring the convergence, the printout 
  includes the split Rhat and the effective sample size. 
}

\value{
  A summary given as a 2 dimension array for the input samples: each row is
  for one parameter; the columns are the mean, standard deviation, quantiles,
  split Rhat, the effective sample size, etc.  
}

\references{
  Stan Development Team (2013) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org}. 
}

\seealso{
  S4 class \code{\linkS4class{stanfit}} and particularly its method
  \code{print}.  
}

\examples{
ecode <- '
  parameters {
    real y[2];
  } 
  model {
    y[1] ~ normal(0, 1);
    y[2] ~ double_exponential(0, 1);
  }
'
fit <- stan(model_code = ecode)
monitor(extract(fit, permuted = FALSE, inc_warmup = TRUE))
} 
