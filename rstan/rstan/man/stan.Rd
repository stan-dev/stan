\name{stan}
\alias{stan}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit a Stan model
%%  ~~function to do ... ~~
}
\description{
Fit a model in Stan modeling language and return 
fitted result. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
stan(file, model.name = "anon_model", model.code = "", 
  data = list(), pars = NA, n.chains = 1L,
  n.iter = 2000L, n.warmup = floor(n.iter/2), n.thin = 1L, init.t = "random",
  init.v = NULL, seed = sample.int(.Machine$integer.max, 1), 
  sample.file, verbose = FALSE, ..., boost.lib = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{file}{The file specifying the model in Stan language.}
  \item{model.name}{A character for naming the model.}
  \item{model.code}{A character string specifying the model.  An alternative
    way for specifying model instead of \code{file}.}
  \item{data}{An object of class \code{list} or \code{environment}
    providing the data for the model.}    
  \item{pars}{A vector of character string specifying parameters of interests; defaults
    to \code{NA} indicating all parameters in the model.}
  \item{n.chains}{A positive integer specifying number of chains; defaults to 1.} 
  \item{n.iter}{A positive integer specifying how many iterations for all the
    chains (including warmup). The default is 2000.}
  \item{n.warmup}{A positive integer specifying number of initial samples to be
    discarded. The default is \code{n.iter / 2}. Note that this is also known as burnin.}
  \item{n.thin}{A positive integer specifying the period for saving samples; defaults to 1.} 
  \item{seed}{The seed for random number generation. The default is generated from 1 to 
    the maximum integer supported by R on the machine. Note that when multiple chains
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
  \item{sample.file}{A character string of file name for specifying where the 
    samples are written into. If not provided, files are not created. Note 
    that when the folder specified is not writable, \code{tempdir()} is used. 
    And when there are multiple chains, an underscore and chain number is appended
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

    More details could be 
    found in Stan's manual. Note that by default, 
    \code{leapfrog_steps} is \code{-1} and 
    \code{equal_step_sizes} is \code{FALSE} so that
    NUTS2 sampler, the default sampler one, is used.

    In addition, \code{refresh} (\code{integer}) could be used to
    control how to indicate the progress during sampling (i.e. 
    show the progress every \code{refresh} iterations).} 

  \item{boost.lib}{The path for boost library that can be downloaded from
     \url{http://http://www.boost.org/users/download/}. This provides an
     alternative path for the boost library used in Stan.}
}
\details{
  \code{stan} is a one step function to fit a 
  Stan model. It first compiles the model represented by
  Stan modeling language to C++ code. Second, the 
  C++ code is compiled into binary shared object, which
  is loaded into the current \R session (an object
  of S4 class \code{stanmodel} is created). Then samples 
  are drawn and wrapped in an object of S4 class \code{stanfit}, 
  which provides functions such as \code{print}, \code{summary}, and \code{plot}
  to take a look of the results of the fitted model. 
}
\value{
  Fitted results as an object of S4 class \code{stanfit}.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
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
  \code{\link{sampling}} 
}
\examples{
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
  s.f <- stan(model.code = scode, verbose = FALSE) 
  print(s.f)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{rstan} 
