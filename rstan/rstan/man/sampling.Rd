\name{sampling-methods}
\docType{methods}
\alias{sampling} 
\alias{sampling-methods}
\alias{sampling,stanmodel-method}

\title{\code{sampling}: draw samples from Stan model}
\description{
  Draw samples from the model defined by class \code{stanmodel}. 
  This method is a generic function of the S4 class \code{stanmodel}. 
}

\usage{
  %% sampling(object, \dots)  
  \S4method{sampling}{stanmodel}(object, data = list(), pars = NA, 
    chains = 4, iter = 2000, warmup = floor(iter/2), thin = 1,
    seed = sample.int(.Machine$integer.max, 1), init = 'random', 
    check_data = TRUE, sample_file, verbose = FALSE, \ldots)   
} 

\section{Methods}{
  \describe{
    \item{sampling}{\code{signature(object = "stanmodel")}}{
      %%  ~~describe this method here~~
      Call sampler (NUTS or HMC, depending on parameters) 
      to draw samples from the model defined by S4 class \code{stanmodel} 
      given the data, initial values, etc. 
    }
  }
}

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanmodel}}.}

  \item{data}{An object of class \code{list}, \code{environment}
    providing the data for the model, or a vector character strings 
    for all the names of objects used as data in the working space. 
    See the notes in \code{\link{stan}}.} 

  \item{pars}{A vector of character strings specifying parameters of interest; defaults
    to \code{NA} indicating all parameters in the model. Only samples for parameters given 
    in \code{pars} are stored in the fitted results.}

  \item{chains}{A positive integer specifying number of chains; defaults to 4.} 
 
  \item{iter}{A positive integer specifying how many iterations for each chain
    (including warmup). The default is 2000.}

  \item{warmup}{A positive integer specifying the number of warmup (aks burnin)
    iterations.  If step-size adaptation is on (which it is by default), this 
    also controls the number of iterations for which adaptation is run (and
    hence the samples should not be used for inference). The default is \code{iter/2}.}

  \item{thin}{A positive integer specifying the period for saving samples; defaults to 1.} 

  \item{seed}{The seed for random number generation. The default is generated from 1 to 
    the maximum integer supported by \R on the machine. Even if multiple chains
    are used, only one seed is needed, with other chains having seeds  
    derived from that of the first chain to avoid dependent samples.}

  \item{init}{One of digit \code{0}, string \code{"0"} or \code{"random"}, 
    a function that returns a list, or a list of initial parameter values with
    which to indicate how the initial values of parameters are specified. 
    \code{"0"}: initialize all to be zero on the unconstrained support; 
    \code{"random"}: randomly generated;
    \code{list}: a list of lists equal in length to the number of chains 
     (parameter \code{chains}), where each list in the list of lists specifies the initial
      values of parameters by name for the corresponding chain.
    \code{function}: a function that returns a list for specifying 
      the initial values of parameters for a chain. The function
      can take an optional parameter \code{chain_id}. }

  \item{check_data}{Logical: if \code{TRUE}, the data would be preprocessed; 
    otherwise not. If the data is not checked and preprocessed, it is safe 
    to leave it to be the default \code{TRUE}. See the notes in \code{\link{stan}}.}

  \item{sample_file}{A character string of file name for specifying where to 
    write samples for \emph{all} parameters and other saved quantities. 
    If not provided, files are not created. When the folder specified 
    is not writable, \code{tempdir()} is used. 
    When there are multiple chains, an underscore and chain number are appended
    to the file name.}
   
  \item{verbose}{\code{TRUE} or \code{FALSE}: flag indicating whether 
     to print intermediate output from Stan on the console, which might
     be helpful for model debugging.}

  \item{\dots}{Other parameters for the sampler: 
    \enumerate{
      \item \code{leapfrog_steps} (\code{integer})
      \item \code{epsilon} (\code{double})
      \item \code{gamma} (\code{double}) 
      \item \code{delta} (\code{double})
      \item \code{equal_step_sizes} (\code{TRUE} or \code{FALSE})
      \item \code{max_treedepth} (\code{integer})
    }

    More details can be 
    found in Stan's manual. By default, 
    \code{leapfrog_steps} is \code{-1} and 
    \code{equal_step_sizes} is \code{FALSE} so that
    NUTS2 sampler, the default sampler, is used.

    In addition, \code{refresh} (\code{integer}) can be used to
    control how to indicate the progress during sampling 
    (i.e. show the progress every \code{refresh} iterations).
    By default, \code{refresh = max(iter/10, 1)}. 
    The progress indicator is turned off if \code{refresh < 0}.

    Another parameter is \code{test_grad} (\code{TRUE} or \code{FALSE}).  
    If \code{test_grad=TRUE}, Stan will not do any sampling. Instead,
    the gradient calculation is tested and printed out and the fitted
    \code{stanfit} object is in test gradient mode.  By default, it is
    \code{FALSE}. 
  }
}
\value{
   An object of S4 class \code{stanfit} representing
   the fitted results. 
} 

\seealso{
  \code{\linkS4class{stanmodel}}, \code{\linkS4class{stanfit}} 
}

\examples{\dontrun{
m <- stan_model(model_code = 'parameters {real y;} model {y ~ normal(0,1);}')
f <- sampling(m, iter = 100)
}} 

\keyword{methods}
