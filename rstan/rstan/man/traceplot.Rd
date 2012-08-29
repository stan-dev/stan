\name{traceplot-methods}
\docType{methods}
\alias{traceplot} 
\alias{traceplot,stanfit-method}

\title{\code{traceplot}: draw the traces of the sample}
\description{Draw the traces of sample of individual chains, a way to check the convergence by vision.}

\usage{
  %% traceplot(object, \dots)  
  \S4method{traceplot}{stanfit}(object, pars, inc_warmup = TRUE, ask = FALSE) 
} 

\section{Methods}{
  \describe{
    \item{traceplot}{\code{signature(object = "stanfit")}}{ Plot the traces of sample for all chains.}
  }
}

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}} object.}
  \item{pars}{A vector of character string specifying parameters to be plotted.} 
  \item{inc_warmup}{\code{TRUE} or \code{FALSE}, indicating whether 
    the warmup sample are included in the trace plot; defaults to \code{TRUE}}
  \item{ask}{\code{TRUE} or \code{FALSE}, to control (for the current device)
    whether the user is prompted before starting a new page of output in the case
    there are a lot of parameters (see \code{\link[grDevices]{devAskNewPage}}).}
}
\value{\code{NULL}} 

\seealso{
  \code{\link[grDevices]{devAskNewPage}} 
}

\examples{\dontrun{
library(rstan) 
fit <- stan(model_code = "parameters {real y;} model {y ~ normal(0,1);}") 
traceplot(fit) 
}} 
\keyword{methods}
