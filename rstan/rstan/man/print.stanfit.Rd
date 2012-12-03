
\name{print}
\alias{print.stanfit}

\title{Print a summary for a fitted model represented by a \code{stanfit} object}
\description{Print basic information regarding the fitted model and 
  a summary for the parameters of interest estimated by the samples included
  in a \code{stanfit} object.
} 

\usage{
  \method{print}{stanfit}(x, pars = x@sim$pars_oi, 
      probs = c(0.025, 0.25, 0.5, 0.75, 0.975),
      digits_summary = 1, ...)
}

\arguments{
  \item{x}{An object of S4 class \code{stanfit}}
  \item{pars}{Parameters in which the summaries are interest; defaults to all parameters
    for which samples are saved}
  \item{probs}{Quantiles of interest; defaults to \code{"c(0.025,0.25,0.5,0.75,0.975)"}}
  \item{digits_summary}{The number of significant digits for printing out
    the summary; defaults to 1.} 
  \item{\dots}{passed to method \code{summary} of \code{stanfit}}
}

\details{
  The information regarding the fitted model includes the number of iterations,
  the number of chains, the number of iterations that are saved in the
  \code{stanfit} object (including warmup); which sampler of NUTS1, NUTS2, HMC is
  used; and when the sampling is finished. 

  The summary about parameters includes the mean, the standard error of the
  mean (se_mean), the standard deviation (sd), quantiles, the effective sample
  size (n_eff), and the split Rhat. The summary is computed based on merging 
  samples without warmup iterations from all chains.  

  In addition to parameters, the log-posterior (\code{lp__}) is also treated
  like a parameter in the printout. 
}

\seealso{
  S4 class \code{\linkS4class{stanfit}} and particularly its method
  \code{summary}, which is used to obtain the values that are printed out.
} 

