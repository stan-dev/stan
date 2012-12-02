
\name{print}
\alias{print.stanfit}

\title{Print a summary of parameters estimated in a \code{stanfit} object}
\description{Print basic information regarding the fitted model and 
  the summaries such as mean, standard error, quantiles, etc. for 
  the parameters of interest estimated in the fitted model represented 
  by \code{stanfit} object}

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

\seealso{
  S4 class \code{\linkS4class{stanfit}} and particularly its method \code{summary},
  which is used to obtain the printed out values for parameters. 
} 

