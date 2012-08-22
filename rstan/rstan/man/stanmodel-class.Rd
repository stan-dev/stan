
\name{stanmodel-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanmodel-class}
%% \alias{sampling,stanmodel-method}
\alias{show,stanmodel-method}

\title{Class \code{"stanmodel"}}

\description{
  A \code{stanmodel} object represents the model compiled from C++ code. 
  Mainly, \code{sampling} method for this class are defined to draw
  sample from this model. 
} 

\section{Objects from the Class}{
  Objects are usually created by calling function \code{stan.model}
}


\section{Slots}{
  \describe{
    \item{\code{model.name}:}{The model name, object of \code{character}.} 
    \item{\code{model.code}:}{The model code, object of \code{character}, in Stan modeling language.} 
    \item{\code{.modelmod}:}{Object of \code{environment}  
      that includes a C++ object exposed by package \pkg{Rcpp}, a module mainly for drawing
      sample for the model.}
    \item{\code{dso}:}{Object of S4 class \code{cxxdso}. The container for the dynamic 
      shared objects compiled the C++ code of the model, returned from function
      \code{cxxfunction} in package \pkg{inline}.}
  }
}

\section{Methods}{
  \describe{
    \item{show}{\code{signature(object = "stanmodel")}: print the model in
      Stan modeling language code.}
    \item{sampling}{\code{signature(object = "stanmodel")}: draw sample for
      the model (see \code{\link{sampling}}).}
  }
}

\note{
  Objects of class \code{stanfit} can be saved for using across 
  \R sessions only if \code{save.dso = TRUE} is used during calling
  functions that create \code{stanfit} objects (e.g., \code{stan}). 
  But definitely even if \code{save.dso = TRUE}, it cannot be used across
  platforms. 
}

\seealso{
\code{\link{stanc}} 
}
\examples{
showClass("stanmodel")
}
\keyword{classes}
