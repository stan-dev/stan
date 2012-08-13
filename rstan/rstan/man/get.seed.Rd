\name{get.seed-methods}
\docType{methods}
\alias{get.seed}
\alias{get.seed-methods}
\alias{get.seed,stanfit-method}
\title{\code{get.seed}: obtain the seed used for sampling} 
\description{
 Obtain the seed used in the random number generation during sampling.} 
\usage{
  \S4method{get.seed}{stanfit}(object) 
} 
\section{Methods}{
  \describe{
    \item{\code{signature(object = "stanfit")}}{
      Obtain the seed used for sampling.  
    }
  }
}
\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
} 
\value{
  A numeric value, the one seed used for sampling all the chains. 
} 

\seealso{  
  \code{\link{stan}}, \code{\link{sampling}}, and  
  \code{\linkS4class{stanfit}} 
}
\keyword{methods}
