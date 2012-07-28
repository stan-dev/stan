
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
    \item{\code{model.code}:}{The model code, object of \code{character}, in Stan model language} 
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

\references{
%%  ~~put references to the literature/web site here~~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{stanc}} 
}
\examples{
showClass("stanmodel")
}
\keyword{classes}
