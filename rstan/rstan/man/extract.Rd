\name{extract-methods}
\docType{methods}
\alias{extract} 
\alias{extract-methods}
\alias{extract,stanfit,character-method}

\title{\code{extract}: extract samples from fitted Stan model}
\description{
  Extract samples drawn for a fitted model defined by class \code{stanfit}. 
}

\usage{
  %% extract(object, pars, \dots)  
  \S4method{extract}{stanfit,character}(object, pars, ...) 
} 

\section{Methods}{
  \describe{
    \item{extract}{\code{signature(object = "stanfit", pars = "character")}}{
       Extract samples for a fitted model (defined by class \code{"stanfit"}).  
    }
  }
}

\arguments{
  \item{object}{An object of class \code{"\linkS4class{stanfit}"}.}

  \item{pars}{An object of class \code{"character"} providing the parameter
              names (or other quantity names) of interest.} 
  \item{...}{FIXME. To be added.} 
}

\value{
  A \code{"list"}, every element of which is the samples for a chain. 
  The samples of a chain is in form of an array. 
} 

\examples{
\dontrun{
}
} 

\keyword{methods}
\keyword{stanfit}
% \keyword{ ~~ other possible keyword(s) ~~ }
