
\name{stanmodel-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanmodel-class}
%% \alias{samples,stanmodel-method}
\alias{show,stanmodel-method}

\title{Class \code{"stanmodel"}}

\description{
  Information for a Stan model, typically, a Bayesian model. In particular,
  methods for drawing samples implemented in C++ and the samples themselves are
  exposed to R as a Rcpp module, which can  be instantiated later. 
} 

\section{Objects from the Class}{
  Objects are usually created by calling function \code{"stan.model"}
}


\section{Slots}{
  \describe{
    \item{\code{model.name}:}{The model name, object of \code{"character"}} 
    \item{\code{.modelmod}:}{Object of \code{"list"}  
      that includes C++ object exposed by Rcpp, a 
      module for drawing samples and conducting analysis on the samples.}
  }
}

\section{Methods}{
  \describe{
    \item{show}{\code{signature(object = "stanmodel")}: prints the class. }
    \item{samples}{\code{signature(object = "stanmodel")}: draw samples from
      the model (see \code{\link{samples}}).}
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
%%  ~~objects to See Also as \code{\link{~~fun~~}}, ~~~
%%  ~~or \code{\linkS4class{CLASSNAME}} for links to other classes ~~~
}
\examples{
showClass("stanmodel")
}
\keyword{classes}
