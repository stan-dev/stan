\name{sampling-methods}
\docType{methods}
\alias{sampling} 
\alias{sampling-methods}
\alias{sampling,stanmodel-method}

\title{\code{sampling}: draws sample from Stan model}
\description{
  Draw sample from the model defined by class \code{stanmodel}. 
  This is a generic function of S4 class \code{stanmodel}. 
}

\usage{
  %% sampling(object, \dots)  
  \S4method{sampling}{stanmodel}(object, data = list(), pars = NA, 
    n.chains = 4L, n.iter = 2000L, n.warmup = floor(n.iter / 2), n.thin = 1L,
    seed = sample.int(.Machine$integer.max, 1), init.t = 'random', init.v = NULL, 
    check.data = TRUE, sample.file, verbose = FALSE, \ldots)   
} 

\section{Methods}{
  \describe{
    \item{sampling}{\code{signature(object = "stanmodel")}}{
      %%  ~~describe this method here~~
      Call samplers such as NUTS1 and NUTS2 to draw sample from the model defined by S4 class \code{stanmodel} 
      given the data, initial values, etc. 
    }
  }
}

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanmodel}} object.}

  \item{data}{An object of class \code{list} or \code{environment}
    providing the data for the model.}    

  \item{pars}{A vector of character string specifying parameters of interest; defaults
    to \code{NA} indicating all parameters in the model. Only sample for parameters given 
    in \code{pars} are stored in the fitted results.}

  \item{n.chains}{A positive integer specifying number of chains; defaults to 1.} 
 
  \item{n.iter}{A positive integer specifying how many iterations for all the
    chains (including warmup). The default is 2000.}

  \item{n.warmup}{A positive integer specifying the number of iterations to be
    discarded. The default is \code{n.iter / 2}. This is also known as burnin.}

  \item{n.thin}{A positive integer specifying the period for saving sample; defaults to 1.} 

  \item{seed}{The seed for random number generation. The default is generated from 1 to 
    the maximum integer supported by \R on the machine. When multiple chains
    are used, only one seed is needed.}

  \item{init.t}{One of \code{0} (or \code{"0"}), \code{"user"}, or \code{"random"} 
    indicating how the initial values of parameters are specified. \code{"0"}: 
    initialize all to be zero on the unconstrained support; 
    \code{"user"}: specified by the user in \code{init.v}; and 
    \code{"random"}: randomly generated.}

  \item{init.v}{A function or a list to specify initial values. When it 
    is a function, it optionally takes a parameter \code{chain.id}. If it 
    is a list, the length of the list must be the number of chains and 
    each element is a named list specifying the initial values for all parameters.}

  \item{check.data}{Logical: if \code{TRUE}, the data would be preprocessed; 
    otherwise not. If the data is not checked and preprocessed, it is safe 
    to leave it to be the default \code{TRUE}.}

  \item{sample.file}{A character string of file name for specifying where the 
    sample are written into. If not provided, files are not created. Note 
    that when the folder specified is not writable, \code{tempdir()} is used. 
    And when there are multiple chains, an underscore and chain number are appended
    to the file name.} 
   
  \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
    intermediate output inside the function, which might be helpful for
    debugging the model, are printed or not.}

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
  }
}
\value{
   An object of S4 class \code{stanfit} that represented 
   the fitted results. 
} 

\seealso{
  \code{\linkS4class{stanmodel}}, \code{\linkS4class{stanfit}} 
}

\examples{\dontrun{
m <- stan.model(model.code = 'parameters {real y;} model {y ~ normal(0,1);}')
f <- sampling(m, n.iter = 100)
}} 

\keyword{methods}
