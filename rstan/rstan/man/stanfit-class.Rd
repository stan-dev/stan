
\name{stanfit-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanfit-class}
\alias{summary,stanfit-method}
\alias{extract,stanfit-method}
\alias{print,stanfit-method}
\alias{show,stanfit-method}

\title{Class \code{"stanfit"}}

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
    \item{\code{model.pars}:}{The names of parameters (or transformed
      parameters, derived quantities), object of \code{"character"}} 
    \item{\code{num.chains}:}{The number of chains drawn for the model, object 
      of \code{"numeric"}} 
    \item{\code{.fit}:}{Object of \code{"list"}  
      that includes an instance of a C++ object, which contains 
      all the samples and provide analyses of samples.}
  }
}

\section{Methods}{
  \describe{
    \item{show}{\code{signature(object = "stanfit")}: prints the summaries
      for the fitted model. }
    \item{summary}{\code{signature(object = "stanfit")}: summarizes the 
      distributions of quantities using the samples: the quantiles 
      (for example, 2.5\%, 50\%, 97.5\%, which could be specified), 
      the mean, the standard deviation (SD), the effective sample size (ESS),
      and the split Rhat (split potential split potential scale reduction).}
    \item{extract}{\code{signature(object = "stanfit", pars = "character")}: 
      get the samples for all chains for specified parameters specified by
      \code{pars}.}
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
showClass("stanfit")
}
\keyword{classes}
