\name{stan.code-methods}
\docType{methods}
\alias{stan.code}
\alias{stan.code-methods}
\alias{stan.code,stanfit-method}
\title{\code{stan.code}: obtain the model's Stan code for fitted model} 
\description{
 Obtain the Stan model language code for the fitted model. 
 This is a generic function of class \code{stanfit}. 
}
\usage{
  \S4method{stan.code}{stanfit}(object, print = FALSE) 
} 
\section{Methods}{
  \describe{
    \item{\code{signature(object = "stanfit")}}{
      Obtain the Stan model language code for a fitted model defined 
      by class \code{stanfit}. 
    }
  }
}
\arguments{
  \item{object}{An object of class \code{"\linkS4class{stanfit}"}.}
  \item{print}{If \code{TRUE}, the model's Stan code will be printed
    out; defaults to \code{FALSE}.}  
} 
\value{
  A character string of the model's code in Stan model language.
} 
\keyword{methods}
\keyword{stanfit} 
