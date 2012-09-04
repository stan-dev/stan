
\name{stanmodel-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanmodel-class}
%% \alias{sampling,stanmodel-method}
\alias{get_cppcode}
\alias{get_cppcode,stanmodel-method}
\alias{show,stanmodel-method}

\title{Class representing model compiled from C++}

\description{
  A \code{stanmodel} object represents the model compiled from C++ code. 
  The \code{sampling} method defined in this class may be used to 
  draw samples from the model.
} 

\section{Objects from the Class}{
  Instances of \code{stanmodel} are usually created by calling function \code{stan_model} or function \code{stan}.
}


\section{Slots}{
  \describe{
    \item{\code{model_name}:}{The model name, an object of type \code{character}.} 
    \item{\code{model_code}:}{The Stan model specification, an object of type \code{character}.} 
    \item{\code{.modelmod}:}{Object of type \code{environment}  
      that includes a C++ object exposed by package \pkg{Rcpp};  
      primarily used for drawing samples for the model.}
    \item{\code{dso}:}{Object of S4 class \code{cxxdso}. The container for the dynamic 
      shared objects compiled from the C++ code of the model, returned from function
      \code{cxxfunction} in package \pkg{inline}.}
  }
}

\section{Methods}{
  \describe{
    \item{\code{show}}{\code{signature(object = "stanmodel")}: print the Stan model specification.}
    \item{\code{sampling}}{\code{signature(object = "stanmodel")}: draw samples for
      the model (see \code{\link{sampling}}).}
    \item{\code{get_cppcode}}{\code{signature(object = "stanmodel")}: return the C++ code for the model
      as a character string. This is part of the C++ code that is compiled to the dynamic 
      shared object for the model.}
  }
}

\note{
  Objects of class \code{stanmodel} can be saved for use across 
  \R sessions only if \code{save_dso = TRUE} is set during calling
  functions that create \code{stanmodel} objects (e.g., \code{stan} 
  and \code{stan_model}). 

  Even if \code{save_dso = TRUE}, the model cannot be loaded on
  a platform (operating system, etc.) that differs from
  the one on which it was compiled.
}

\seealso{
\code{\link{stanc}} 
}
\examples{
showClass("stanmodel")
}
\keyword{classes}
