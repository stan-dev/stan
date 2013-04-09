\name{traceplot-methods}
\docType{methods}
\alias{traceplot} 
\alias{traceplot,stanfit-method}

\title{\code{traceplot}: draw the traces of the sample}
\description{Draw the traceplot corresponding to one or more
 Markov chains, providing a visual way to inspect sampling behavior
 and assess mixing across chains and convergence.}

\usage{
  %% traceplot(object, \dots)  
  \S4method{traceplot}{stanfit}(object, pars, inc_warmup = TRUE, ask = FALSE, ...) 
} 

\section{Methods}{
  \describe{
    \item{traceplot}{\code{signature(object = "stanfit")}}{ Plot the traces of sample for all chains.}
  }
}

\arguments{
  \item{object}{An instance of class \code{\linkS4class{stanfit}}.}
  \item{pars}{A vector of character string specifying the  
      parameters to be plotted.} 
  \item{inc_warmup}{\code{TRUE} or \code{FALSE}, indicating whether 
    the warmup sample are included in the trace plot; defaults to \code{TRUE}}
  \item{ask}{\code{TRUE} or \code{FALSE}, to control (for the current device)
    whether the user is prompted before starting a new page of output in the case
    there are a lot of parameters (see \code{\link[grDevices]{devAskNewPage}}).}
  \item{\dots}{Additional parameters include \code{nrow} and \code{ncol}, default
    to 4 and 2, respectively. When the number of traceplots are larger, the plot
    layout will be \code{nrow * ncol}.}
}
\value{\code{NULL}} 

\seealso{
  \code{\link[grDevices]{devAskNewPage}} 
}

\examples{\dontrun{
library(rstan) 
fit <- stan(model_code = "parameters {real y;} model {y ~ normal(0,1);}") 
traceplot(fit) 
}

# cache of a saved stanfit object 
# the Stan code for this model is in the example of function stan
exfit <- rstan:::exfit  
print(exfit)
traceplot(exfit)
} 
\keyword{methods}
