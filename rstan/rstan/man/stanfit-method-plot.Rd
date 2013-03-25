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
  \S4method{plot}{stanfit,missing}(x, pars, display_parallel = FALSE, 
  ask = TRUE, npars_per_page = 6)
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
  \item{ask}{\code{TRUE} or \code{FALSE}, to control (for the current device)
    whether the user is prompted before starting a new page of output in the case
    there are a lot of parameters (see \code{\link[grDevices]{devAskNewPage}}).}
  \item{npars_per_page}{An integer to specify the number of parameters 
    that would be plotted per page.}
}
\value{\code{NULL}} 
\examples{\dontrun{
library(rstan) 
fit <- stan(model_code = "parameters {real y;} model {y ~ normal(0,1);}") 
plot(fit) 
}} 
\keyword{methods}
