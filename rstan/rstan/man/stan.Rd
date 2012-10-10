\name{stan}
\alias{stan}
\title{Fit a model using Stan}
\description{
  Fit a model defined in the Stan modeling language and  
  return the fitted result as an instance of \code{stanfit}. 
}
\usage{
stan(file, model_name = "anon_model", model_code = "", 
  fit = NA, data = list(), pars = NA, chains = 4,
  iter = 2000, warmup = floor(iter/2), thin = 1, 
  init = "random", seed = sample.int(.Machine$integer.max, 1), 
  sample_file, save_dso = TRUE, 
  verbose = FALSE, ..., 
  boost_lib = NULL,
  eigen_lib = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{file}{A character string file name or a connection that \R supports 
    containing the text of a model specification in the Stan modeling 
    language; a model may also be specified directly
    as a character string using parameter \code{model_code} or through
    a previous fit using parameter \code{fit}. 
    When \code{fit} is specified, parameter \code{file} is ignored.}
  \item{model_name}{A character string naming the model; defaults 
    to \code{"anon_model"}. However, the model name would be derived from 
    \code{file} or \code{model_code} (if \code{model_code} is the name
    of a character string object) if \code{model_name} is not specified.}
  \item{model_code}{A character string either containing the model definition
    or the name of a character string object in the workspace. This 
    parameter is used only if parameter \code{file} is not specified. 
    When \code{fit} is specified, the model compiled previously
    is used so specifying \code{model_code} is ignored.}
  \item{fit}{An instance of S4 class \code{stanfit} derived from
             a previous fit;  defaults to \code{NA}. 
    If \code{fit} is not \code{NA}, the compiled model associated with the fitted result 
    is re-used; thus the time that would otherwise be spent 
    recompiling the C++ code for the model can be saved.}
  \item{data}{An object of class \code{list}, \code{environment}
    providing the data for the model, or a vector of character strings 
    for all the names of objects used as data in the working space. 
    See the notes below.}    
  \item{pars}{A vector of character string specifying parameters of interest; defaults
    to \code{NA} indicating all parameters in the model. Only samples for parameters
    given in \code{pars} are stored in the fitted results.}
  \item{chains}{A positive integer specifying number of chains; defaults to 4.} 
  \item{iter}{A positive integer specifying how many iterations for each 
    chain (including warmup). The default is 2000.}
  \item{warmup}{A positive integer specifying number of warmup (aka burnin)
     iterations.  This also specifies the number of iterations used for step-size
     adaptation, so warmup samples should not be used for inference.
     The default is \code{iter/2}.}
  \item{thin}{A positive integer specifying the period for saving sample; defaults to 1.} 
  \item{seed}{The seed for random number generation. The default is generated from 1 to 
    the maximum integer supported by \R on the machine. When multiple chains
    are used, only one seed is needed, with other chains' seeds being
    generated from the first chain's seed to prevent dependency among the
    random number streams for the chains.}
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
      can take an optional parameter \code{chain_id}. 
  }
  \item{sample_file}{A character string of file name for specifying where to 
    write samples for \emph{all} parameters and other saved quantities. 
    If not provided, files are not created. When the folder specified 
    is not writable, \code{tempdir()} is used. 
    When there are multiple chains, an underscore and chain number are appended
    to the file name.}

  \item{save_dso}{Logical, with default \code{TRUE}, indicating whether the 
    dynamic shared object (DSO) compiled from the C++ code for the model 
    will be saved or not. If \code{TRUE}, we can draw samples from
    the same model in another \R session using the saved DSO (i.e., 
    without compiling the C++ code again).
    This parameter only takes effect if \code{fit} is not used; with  
    \code{fit} defined, the DSO from the previous run is used.
    When \code{save_dso=TRUE}, the fitted object can be loaded from 
    what is saved previously and used from sampling, if the compiling is
    done on the same platform, that is, same operating system and same 
    architecture (32bits or 64bits). 
}

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
    control how to indicate the progress during sampling (i.e. 
    show the progress every \code{refresh} iterations). 
    By default, \code{refresh = max(iter/10, 1)}. 
    The progress indicator is turned off if \code{refresh < 0}.

    Another parameter is \code{test_grad} (\code{TRUE} or \code{FALSE}).  
    If \code{test_grad=TRUE}, Stan will not do any sampling. Instead,
    the gradient calculation is tested and printed out and the fitted
    \code{stanfit} object is in test gradient mode.  By default, it is
    \code{FALSE}. 
 
    When a new model is fitted starting from Stan model code, 
    \code{\dots} is passed to \code{stan_model} and thus \code{stanc}.
  } 

  \item{boost_lib}{The path for an alternative version of the Boost C++ 
     library to the one supplied with RStan.}

  \item{eigen_lib}{The path for an alternative version of the Eigen C++ 
     library to the one in \pkg{RcppEigen}.} 
}
\details{
  \code{stan} does all of the work of fitting a
  Stan model and returning the results as an instance of \code{stanfit}. 
  First, it translates the Stan model to C++ code.  Second, the 
  C++ code is compiled into a binary shared object, which
  is loaded into the current \R session (an object
  of S4 class \code{stanmodel} is created). Finally, sample
  are drawn and wrapped in an object of S4 class \code{stanfit}, 
  which provides functions such as \code{print}, \code{summary}, and \code{plot}
  to inspect and retrieve the results of the fitted model. 

  \code{stan} can also be used to sample again from a fitted model under
  different settings (e.g., different \code{iter}) by providing 
  argument \code{fit}. In this case, the compiled C++ code 
  for the model is reused. 
}
\value{
  Fitted results as an object of S4 class \code{stanfit}. If error occurs 
  before or during sampling, and if \code{test_grad = TRUE}, the returned
  object would not contain samples. But the compiled binary object for the 
  model is still included, so we can reuse the returned object for another
  sampling. 
}

\note{
  The data passed to \code{stan} are preprocessed before
  passing to Stan. This process mainly includes the following:
  \enumerate{
    \item Change the data of type from \code{double} 
      to \code{integer} if no accuracy is lost. The main
      reason is that by default, \R uses \code{double}
      as data type such as in \code{a <- 3}. But Stan 
      will not read data of type \code{int} from \code{real} 
      and it reads data from \code{int} if the data 
      type is declared as \code{real}. 
   
   \item Check if there is \code{NA} in the data. 
     Unlike BUGS, Stan does not allow missing data. Any \code{NA} values
     in supplied data will cause the function to stop and report an error.
  
   \item Check data types passed in.  Stan allows only numeric data, that is,
     doubles, integers, and arrays of these.  Data of other types (for
     examples, characters) are not passed to Stan. Data of  
     \code{data.frame} are converted to \code{matrix} by
     function \code{data.matrix}. 
     
   \item Check whether there are objects in the data list with duplicated names.
     Duplicated names, if found, will cause the function to stop and
     report an error. 

   \item Check whether the names of objects in the data list are legal 
     Stan names. If illegal names are found, it will stop and 
     report an error. See Stan's manul for the rules of variable names. 
  } 

  The returned S4 class \code{stanfit} will save the compiled model
  across sessions if the option \code{save_dso = TRUE}
  is supplied.  

  The function accepts a previously fitted instance of \code{stanfit}
  through parameter \code{fit} in order to reuse the compiled model; other
  configuration may change.

  The optimization level for compiling the C++ code generated for the 
  model can be set by \code{set_cppo}. In general, the higher the optimization
  level is set, the faster the generated binary code for the model runs. However, 
  the binary code generated for the model runs fast by using higher
  optimization level is at the cost of long time to compile the C++ code. 
}

\references{
  Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org}. 
}
\seealso{
  \code{\link{stanc}} for translating model code in Stan modeling language to C++, 
  \code{\link{sampling}} for sampling, and \code{\linkS4class{stanfit}} for the fitted results.

  see \code{\link{set_cppo}} for setting higher optimization level for compiling the C++ 
  code.
  
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
fit1 <- stan(model_code = scode, iter = 10, verbose = FALSE) 
print(fit1)
fit2 <- stan(fit = fit1, iter = 10000, verbose = FALSE) 
}} 
\keyword{rstan} 
