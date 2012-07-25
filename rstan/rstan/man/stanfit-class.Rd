
\name{stanfit-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanfit-class}
\alias{summary,stanfit-method}
%% \alias{extract,stanfit-method}
\alias{print,stanfit-method}
\alias{show,stanfit-method}
\alias{plot,stanfit-method}
%% \alias{traceplot,stanfit-method}

\title{Class \code{"stanfit"}}

\description{
  Samples and other inferences information for a Stan model, typically, a
  Bayesian model. \code{print} and \code{plot} are provided for the 
  summaries of the fitted results. 
} 

\section{Objects from the Class}{
  Objects are usually created by calling \code{sampling} function in \code{"stan.model"},
  or by the \code{stan} function, which wraps the \code{sampling} function inside.  
}


\section{Slots}{
  \describe{
    \item{\code{model.name}:}{The model name, object of \code{"character"}} 
    \item{\code{model.pars}:}{The names of parameters (or transformed
      parameters, derived quantities), object of \code{"character"}} 
    \item{\code{par.dims}:}{The dimensions for all parameters, object of \code{"list"}}
    \item{\code{sim}:}{Simulation results including samples for the model, object 
      of \code{"list"}} 
    \item{\code{arg.lst}:}{The arguments used for sampling all chains, object of \code{"list"}}
    \item{\code{.MISC}:}{Miscellaneous helper information used for the fitted model, 
      object of \code{"environment"}}
  }
}

\section{Methods}{
  \describe{
    \item{show}{\code{signature(object = "stanfit")}: prints a very short summary
      for the model.}
    \item{plot}{\code{signature(x = 'stanfit')}: plot an overview of summaries 
      for all parameters.}
    \item{summary}{\code{signature(object = "stanfit")}: summarizes the 
      distributions of quantities using the samples: the quantiles 
      (for example, 2.5\%, 50\%, 97.5\%, which could be specified), 
      the mean, the standard deviation (SD), the effective sample size (ESS),
      and the split Rhat (split potential split potential scale reduction).}
    \item{extract}{\code{signature(object = "stanfit")}: 
      get the samples for all chains for all (or specified ) parameters. 
      (see \code{\link{extract}}).}
    \item{traceplot}{\code{signature(object = "stanfit")}:
      plot the trace of chains.} 
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
