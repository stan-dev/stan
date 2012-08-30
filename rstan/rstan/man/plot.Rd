\name{plot-methods}
\alias{plot-methods} 
% \alias{plot,ANY,ANY-method}
\alias{plot,stanfit,missing-method}
\docType{methods}

\title{\code{plot}: plot an overview of summaries for the fitted model} 
\description{Drawn an overview of parameter summaries for the fitted model.
  In the overview plot, we also indicate the values of Rhats for
  all parameters of interest using differnt colors. In addition to all the
  parameters, the log-posterior is also plotted. 
} 

\usage{
  \S4method{plot}{stanfit,missing}(x, pars, display_parallel = FALSE)
} 

\section{Methods}{
  \describe{
    \item{plot}{\code{signature(x = "stanfit", y = "missing")}}{ Plot an overview of parameter
      summaries for the fitted model.}
  }
}

\arguments{
  \item{x}{An instance of class \code{\linkS4class{stanfit}}.}
  \item{pars}{A vector of character string specifying the  
    parameters to be plotted. If not specified, all parameters
    are used.} 
  \item{display_parallel}{\code{TRUE} or \code{FALSE}, indicating 
    whether to plot the intervals with one line for each chain
    or one line for all chains. The default is \code{FALSE} so
    that only one interval line is drawn for each scalar parameter.} 
}
\value{\code{NULL}} 
\examples{\dontrun{
library(rstan) 
fit <- stan(model_code = "parameters {real y;} model {y ~ normal(0,1);}") 
plot(fit) 
}} 
\keyword{methods}
