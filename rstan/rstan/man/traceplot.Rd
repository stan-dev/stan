\name{traceplot-methods}
\docType{methods}
\alias{traceplot} 
\alias{traceplot,stanfit-method}

\title{\code{traceplot}: draw the traces of samples}
\description{Draw the traces of samples of each chain, a way to check the convergence by vision.}

\usage{
  %% traceplot(object, \dots)  
  \S4method{traceplot}{stanfit}(object, pars, inc.warmup = TRUE, ask = FALSE) 
} 

\section{Methods}{
  \describe{
    \item{traceplot}{\code{signature(object = "stanfit")}}{Plot the traces of samples of each chain.}
  }
}

\arguments{
  \item{object}{An object of class \code{"\linkS4class{stanfit}"} object.}
  \item{pars}{A vector of character string specifying parameters to be plotted.} 
  \item{inc.warmup}{\code{TRUE} or \code{FALSE}, indicating whether 
    the warmup samples are included in the trace plot; defaults to \code{TRUE}}
  \item{ask}{\code{TRUE} or \code{FALSE}, to control (for the current device)
    whether the user is prompted before starting a new page of output in the case
    there are a lot of parameters (see \code{\link[grDevices]{devAskNewPage}}).}
}
\value{} 

\seealso{
  \code{\link[grDevices]{devAskNewPage}} 
}

\examples{
  \dontrun{}
} 

\keyword{methods}
\keyword{traceplot}
\keyword{stanfit}
