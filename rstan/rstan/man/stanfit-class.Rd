
\name{stanfit-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanfit-class}
\alias{summary,stanfit-method}
%% \alias{extract,stanfit-method}
\alias{print,stanfit-method}
\alias{show,stanfit-method}
\alias{plot,stanfit-method}
%% \alias{stan.code,stanfit-method}
%% \alias{traceplot,stanfit-method}

\title{Class \code{stanfit}: fitted Stan model}

\description{
  Samples and other inferences information for a Stan model, typically, a
  Bayesian model. \code{print} and \code{plot} are provided for the 
  summaries of the fitted results. 
} 

\section{Objects from the Class}{
  Objects are usually created by calling \code{sampling} function in S4 class \code{stan.model},
  or by the \code{stan} function, which wraps the \code{sampling} function inside.  
}


\section{Slots}{
  \describe{
    \item{\code{model.name}:}{The model name, object of \code{character}} 
    \item{\code{model.pars}:}{The names of parameters (or transformed
      parameters, derived quantities), object of \code{character}} 
    \item{\code{par.dims}:}{The dimensions for all parameters, object of \code{list}}
    \item{\code{sim}:}{Simulation results including samples for the model, object 
      of \code{list}} 
    \item{\code{inits}:}{The initial values either specified or generated
      randomly for all chains, object of \code{list}. Each member is a list for a chain.} 
    \item{\code{stan.args}:}{The arguments used for sampling all chains, object of \code{list}}
    \item{\code{.MISC}:}{Miscellaneous helper information used for the fitted model, 
      object of \code{environment}}
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
    \item{stan.code}{\code{signature(object = "stanfit")}:
      get the Stan code for the fitted model.}
    \item{get.inits}{\code{signature(object = "stanfit")}:
      get the initial values for parameters used in sampling all chains.} 
    \item{get.seed}{\code{signature(object = "stanfit")}: 
      get the seed that used for sampling.} 
  }
}

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://code.google.com/p/stan}. 
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
  \code{\link{stan}} and \code{\linkS4class{stanmodel}} 
}
\examples{
  showClass("stanfit")
}
\keyword{classes}
\keyword{stanfit} 
