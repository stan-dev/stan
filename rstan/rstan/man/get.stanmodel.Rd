\name{get.stanmodel-methods}
\docType{methods}
\alias{get.stanmodel}
\alias{get.stanmodel-methods}
\alias{get.stanmodel,stanfit-method}
\title{\code{get.stanmodel}: obtain the S4 class \code{stanmodel} from the fitted model} 
\description{
 An object of S4 class \code{stanfit} is the fitted results (mainly 
 sample) from a \code{stanmodel}. This method is used to extract
 the S4 class \code{stanmodel} object from the fitted model.} 
\usage{
  \S4method{get.stanmodel}{stanfit}(object) 
} 
\section{Methods}{
  \describe{
    \item{\code{signature(object = "stanfit")}}{
      Extract the \code{stanmodel} object. 
    }
  }
}
\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
} 
\value{
  An object of class \code{\linkS4class{stanmodel}}. 
} 

\seealso{  
  \code{\linkS4class{stanfit}} and \code{\linkS4class{stanmodel}} 
}
\keyword{methods}
