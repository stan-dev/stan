
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
  samples from this model. 
} 

\section{Objects from the Class}{
  Objects are usually created by calling function \code{stan.model}
}


\section{Slots}{
  \describe{
    \item{\code{model.name}:}{The model name, object of \code{character}} 
    \item{\code{model.code}:}{The model code, object of \code{character}, in Stan modeling language} 
    \item{\code{.modelmod}:}{Object of \code{list}  
      that includes C++ object (just the definition, not an instance) exposed
      by Rcpp, a module mainly for drawing samples for the model.}
  }
}

\section{Methods}{
  \describe{
    \item{show}{\code{signature(object = "stanmodel")}: prints the class. }
    \item{sampling}{\code{signature(object = "stanmodel")}: draw samples for
      the model (see \code{\link{sampling}}).}
  }
}

\note{
  Objects of class \code{stanfit} could \strong{not} be saved as ordinary 
  R objects. One reason is that the C++ code for the model is
  compiled and loaded all temporarily using package inline, which
  does not provide any approach to save and reload the compiled binary
  objects.  
}

\seealso{
\code{\link{stanc}} 
}
\examples{
showClass("stanmodel")
}
\keyword{classes}
