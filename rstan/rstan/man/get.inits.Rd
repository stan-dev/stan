\name{get.inits-methods}
\docType{methods}
\alias{get.inits}
\alias{get.inits-methods}
\alias{get.inits,stanfit-method}
\title{\code{get.inits}: obtain the inital values for parameters} 
\description{
 Obtain the inital values for parameters that used for sampling all
 chains. This is a generic function of S4 class \code{stanfit}. 
}
\usage{
  \S4method{get.inits}{stanfit}(object) 
} 
\section{Methods}{
  \describe{
    \item{\code{signature(object = "stanfit")}}{
      Obtain the inital values for parameters that used for sampling all
      chains. 
    }
  }
}
\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
} 
\value{
  A list with each element being a named list for the inital values
  used for sampling each chain. 
} 
\seealso{
  \code{\link{stan}}, \code{\link{sampling}}, and  
  \code{\linkS4class{stanfit}} 
} 
\keyword{methods}
\keyword{stanfit} 
