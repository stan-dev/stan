\name{stan}
\alias{stan}
\title{Fit a model using Stan}
\description{
  Fit a model in Stan modeling language and return the fitted result. 
}
\usage{
stan(file, model_name = "anon_model", model_code = "", 
  fit = NA, data = list(), pars = NA, n_chains = 4L,
  iter = 2000L, warmup = floor(iter/2), thin = 1L, 
  init = "random", seed = sample.int(.Machine$integer.max, 1), 
  sample_file, save_dso = TRUE, 
  verbose = FALSE, ..., boost_lib = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{file}{A character string or a connection that \R supports to specify the model 
    in Stan modeling language.}
  \item{model_name}{A character for naming the model.}
  \item{model_code}{A character string for specifying the model.  An alternative
    way to specify model instead of using parameter \code{file}.}
  \item{fit}{An instance of S4 class \code{stanfit}; defaults to \code{NA}. 
    When it is not \code{NA}, the model from the fitted result 
    is used to sample. Hence time for recompiling the C++ code
    is saved.} 
  \item{data}{An object of class \code{list}, \code{environment}
    providing the data for the model, or character strings 
    for all the names of objects used as data in the working space.}    
  \item{pars}{A vector of character string specifying parameters of interest; defaults
    to \code{NA} indicating all parameters in the model. Only samples for parameters
    given in \code{pars} are stored in the fitted results.}
  \item{n_chains}{A positive integer specifying number of chains; defaults to 4.} 
  \item{iter}{A positive integer specifying how many iterations for all the
    chains (including warmup). The default is 2000.}
  \item{warmup}{A positive integer specifying number of iterations to be
    discarded. The default is \code{iter / 2}. This is also known as burnin.}
  \item{thin}{A positive integer specifying the period for saving sample; defaults to 1.} 
  \item{seed}{The seed for random number generation. The default is generated from 1 to 
    the maximum integer supported by \R on the machine. When multiple chains
    are used, only one seed is needed.}
  \item{init}{One of \code{0} (or \code{"0"}), \code{"random"}, or a list 
    indicating how the initial values of parameters are specified. \code{"0"}: 
    initialize all to be zero on the unconstrained support; 
    \code{"random"}: randomly generated;
    object of \code{list}: a list of named list. If a list
    is used to specify the initial values, the length of the list should 
    be the number of chains (\code{n\_chains}). And each list 
    is a named list, every element of which specifies the initial 
    value(s) for a parameter.}
  \item{sample_file}{A character string of file name for specifying where the 
    sample for \emph{all} parameters are written into. If not provided, files
    are not created. When the folder specified is not writable, \code{tempdir()} is used. 
    And when there are multiple chains, an underscore and chain number is appended
    to the file name.} 
  \item{save_dso}{Logical with the default of \code{TRUE}: indication of whether the 
    dynamic shared object (DSO) compiled from the C++ code for the model 
    will be saved or not. If \code{TRUE}, we can draw sample from
    the same model in another \R session using the saved DSO (i.e., 
    without compiling the C++ code again).
    This parameter only takes effect when \code{stan} is used to 
    fit a model from scratch, that is, not with specifying parameter \code{fit}.}
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
    control how to indicate the progress during sampling (i.e. 
    show the progress every \code{refresh} iterations).} 

  \item{boost_lib}{The path for boost library that can be downloaded from
     \url{http://www.boost.org/users/download/}. This provides an
     alternative way to specify the boost library used in Stan.}
}
\details{
  \code{stan} is a one step function to fit a 
  Stan model. It first compiles the model represented by
  Stan modeling language to C++ code. Second, the 
  C++ code is compiled into binary shared object, which
  is loaded into the current \R session (an object
  of S4 class \code{stanmodel} is created). Then sample 
  are drawn and wrapped in an object of S4 class \code{stanfit}, 
  which provides functions such as \code{print}, \code{summary}, and \code{plot}
  to take a look of the results of the fitted model. 

  \code{stan} can also be used to sample again from a fitted model under
  different settings (e.g., different \code{iter}) by providing 
  argument \code{fit}. In this case, the compiled C++ code 
  for the model is reused. 
}
\value{
  Fitted results as an object of S4 class \code{stanfit}.
}

\note{
  As Stan does not allow missing data. The data are pre-processed 
  to check if there is any \code{NA} and it would stop when there is. 
  Also, Stan allows only numeric data, that is, doubles and integers. 
  Data of other types (for examples, characters) are not passed to 
  Stan. 

  We can use a previous fitted model object (i.e., an object
  of S4 class \code{stanfit}) as input for function \code{stan} to 
  refit the model, for example, with different configuration or data. 
  However, when an object of class \code{stanfit} is created by 
  for example, calling \code{stan} with \code{save_dso = FALSE},
  the fitted results cannot be used across \R sessions.  
}

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://mc-stan.org}. 
}
\seealso{
  \code{\link{stanc}} compiling model code in Stan modeling language to C++, 
  \code{\link{sampling}} and \linkS4class{stanfit}
}
\examples{\dontrun{
library(rstan)
scode <- "
parameters {
  real y[2]; 
} 
model {
  y[1] ~ normal(0, 1);
  y[2] ~ double_exponential(0, 2);
} 
"
s.f <- stan(model_code = scode, verbose = FALSE) 
print(s.f)
s.f2 <- stan(fit = s.f, iter = 1000, verbose = FALSE) 
}} 
\keyword{rstan} 
