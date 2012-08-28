
\name{stanfit-class} 
\Rdversion{1.1}

\docType{class}
\alias{stanfit-class}
\alias{summary,stanfit-method}
%% \alias{extract,stanfit-method}
\alias{print,stanfit-method}
\alias{show,stanfit-method}
\alias{plot,stanfit-method}
\alias{get_logposterior} 
\alias{get_logposterior,ANY-method}
\alias{get_logposterior,stanfit,logical-method}
\alias{get_adaptation_info} 
\alias{get_adaptation_info,stanfit-method}
\alias{get_adaptation_info,ANY-method}
\alias{get_sampler_params}
\alias{get_sampler_params,ANY-method}
\alias{get_sampler_params,stanfit,logical-method}
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
    \item{\code{sim}:}{Simulation results including sample for the model, object 
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
    \item{show}{\code{signature(object = "stanfit")}: print the default summary
      for the model.}
    \item{plot}{\code{signature(x = 'stanfit')}: plot an overview of summaries 
      for all parameters.}
    \item{print}{\code{signature(x = 'stanfit')}: print the summary of the 
      model with inference about mean (\code{mean}), standard error of the 
      mean (\code{se_mean}), standard deviation (\code{sd}), and quantities
      about quantities of interest based on all chains merged. 
      Most of the printout are the results from function \code{summary}.} 
    \item{summary}{\code{signature(object = "stanfit")}: summarizes the 
      distributions of quantities using the sample: the quantiles 
      (for example, 2.5\%, 50\%, 97.5\%, which can be specified), 
      the mean, the standard deviation (sd), the effective sample size (n_eff),
      and the split Rhat (split potential split potential scale reduction).
      Returned is a named list with elements \code{summary} and \code{c.summary}, 
      summaries for all chains merged without warmup and individual chains
      respectively. For the summary of all chains merged, we also have \code{se_mean},
      the standard error of the mean. 
    }
    \item{extract}{\code{signature(object = "stanfit")}: 
      get the sample for all chains for all (or specified ) parameters. 
      (see \code{\link{extract}}).}
    \item{traceplot}{\code{signature(object = "stanfit")}:
      plot the trace of chains.} 
    \item{get.stancode}{\code{signature(object = "stanfit")}:
      get the Stan code for the fitted model.}
    \item{get.stanmodel}{\code{signature(object = "stanfit")}: 
      get the S4 class \code{stanmodel} from the fitted 
      model.} 
    \item{get.inits}{\code{signature(object = "stanfit")}:
      get the initial values for parameters used in sampling all chains.} 
    \item{get.seed}{\code{signature(object = "stanfit")}: 
      get the seed that used for sampling.} 
    \item{get_logposterior}{\code{signature(object = 'stanfit', 
      inc_warmup = 'logical')}:
      get the log-posterior (up to a constant) for all chains. 
      Each element of the returned \code{list} is the log-posterior 
      for a chain. \code{inc_warmup} indicates whether to include
      the warmup period.} 
    \item{get_adaptation_info}{\code{signature(object = 'stanfit')}:
      obtain the adaptation information for 
      sampler, which now only NUTS2 has. The results are returned 
      as a list, each element of which is a character string
      for a chain. } 
    \item{get_sampler_params}{\code{signature(obtain = 'stanfit', 
      inc_warmup = 'logical')}:
      obtain the parameters used for the sampler such as 
      \emph{stepsize} and \empth{treedepth}. The results are returned
      as a list, each element of which is an array for a chain.
      The array has number of columns corresponding to the number
      of parameters used in the sampler and its column names 
      provide the parameter names.}
  }
}

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://mc-stan.org}. 
}
\seealso{
  \code{\link{stan}} and \code{\linkS4class{stanmodel}} 
}
\examples{
showClass("stanfit")
}
\keyword{classes}
