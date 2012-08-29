\name{extract-methods}
\docType{methods}
\alias{extract} 
\alias{extract-methods}
\alias{extract,stanfit-method}
%\alias{extract,stanfit,character-method}

\title{\code{extract}: extract sample from fitted Stan model}
\description{
  Extract samples drawn from a fitted model represented by an 
  instance of class \code{stanfit}.
}

\usage{
  %% extract(object, \dots)  
  \S4method{extract}{stanfit}(object, pars, permuted = FALSE, inc_warmup = TRUE) 
} 

\section{Methods}{
  \describe{
    \item{extract}{\code{signature(object = stanfit}}{
       Extract samples for a fitted model defined by class \code{stanfit}.  
    }
  }
}

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}}.
  \item{pars}{An object of class \code{"character"} providing the parameter
    names (or other quantity names) of interest. If not specified,
    all parameters and other quantities are used.} 
  \item{permuted}{\code{TRUE} of \code{FALSE}. If \code{TRUE}, all kept 
    samples (that is, warmup sample
    are discarded) in each chain are permuted are merged. If \code{FALSE}, 
    the original order are kept.}
  \item{inc_warmup}{\code{TRUE} of \code{FALSE}. The argument matters
    only if \code{permuted} is \code{FALSE}. If \code{TRUE}, warmup
    sample are included; otherwise excluded.}
}
\value{
  When \code{permuted = TRUE}, this function returns a list,  
  every element of which is an array representing samples from a chain.
  When \code{permuted = FALSE}, an array is returned; the first
  dimension is the iterations; the second is the number of chains; the 
  third are the parameters. Vectors and arrays are expanded to one 
  parameter per cell, with names indicating dimensions.}

% \examples{
%   \dontrun{}
% } 

\keyword{methods}
