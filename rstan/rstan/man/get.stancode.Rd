\name{get.stancode-methods}
\docType{methods}
\alias{get.stancode}
\alias{get.stancode-methods}
\alias{get.stancode,stanfit-method}
\title{\code{get.stancode}: obtain the Stan modeling language code for a fitted model} 
\description{
 Obtain the Stan modeling language code for a fitted model. 
 This is a generic function of S4 class \code{stanfit}. 
}
\usage{
  \S4method{get.stancode}{stanfit}(object, print = FALSE) 
} 
\section{Methods}{
  \describe{
    \item{\code{signature(object = "stanfit")}}{
      Obtain the Stan modeling language code for a fitted model defined 
      by class \code{stanfit}. 
    }
  }
}
\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
  \item{print}{If \code{TRUE}, the model's Stan code will be printed
    out; defaults to \code{FALSE}.}  
} 
\value{
  A character string of the Stan modeling language code.
} 
\seealso{
  \code{stan}, \code{stanc}, and \code{\linkS4class{stanfit}} 
} 
\keyword{methods}
