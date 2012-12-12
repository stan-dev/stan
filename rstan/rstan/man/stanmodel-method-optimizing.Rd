\name{optimizing-methods}
\docType{methods}
\alias{optimizing} 
\alias{optimizing-methods}
\alias{optimizing,stanmodel-method}

\title{\code{optimizing}: obtain a point estimate by maximizing the joint posterior}
\description{
  Obtain a point estimate by maximizing the joint posterior 
  from the model defined by class \code{stanmodel}. 
  This method is a generic function of the S4 class \code{stanmodel}. 
}

\usage{
  %% optimizing(object, \dots)  
  \S4method{optimizing}{stanmodel}(object, data = list(), 
    seed = sample.int(.Machine$integer.max, 1), init = 'random', 
    check_data = TRUE, sample_file, verbose = FALSE, \dots)   
} 

\section{Methods}{
  \describe{
    \item{optimizing}{\code{signature(object = "stanmodel")}}{
      %%  ~~describe this method here~~
      Call Stan's optimization method to obtain a point estimate 
      for the model defined by S4 class \code{stanmodel} 
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
  \item{seed}{The seed for random number generation. The default is generated
    from 1 to the maximum integer supported by \R on the machine. 
    When a seed is specified by a number, \code{as.integer} will be applied to
    it. If \code{as.integer} produces \code{NA}, the seed is generated
    randomly.  We can also specify a seed using a character string of digits,
    such as \code{"12345"}, which will be converted to integer.}
  \item{init}{One of digit \code{0}, string \code{"0"} or \code{"random"}, 
    a function that returns a list, or a named list of initial parameter
    values.
    \code{"0"}: initialize all to be zero on the unconstrained support; 
    \code{"random"}: randomly generated;
    \code{list}: a list specifying the initial values of parameters by name.
    \code{function}: a function that returns a list for specifying 
      the initial values of parameters for a chain.}
  \item{check_data}{Logical: if \code{TRUE}, the data would be preprocessed; 
    otherwise not. If the data is not checked and preprocessed, it is safe 
    to leave it to be the default \code{TRUE}. See the notes in
    \code{\link{stan}}.}
  \item{sample_file}{A character string of file name for specifying where to 
    write samples for \emph{all} parameters and other saved quantities. 
    If not provided, files are not created. When the folder specified 
    is not writable, \code{tempdir()} is used.}
  \item{verbose}{\code{TRUE} or \code{FALSE}: flag indicating whether 
     to print intermediate output from Stan on the console.}
  \item{\dots}{Other optional parameters (none currently).}
}
\value{
   A list with components if the optimization is done successfully:
   \item{par}{The point estimate found.}
   \item{value}{The value of the log-posterior (up to an additive constant, 
   the \code{"lp__"} in Stan) corresponding to \code{par}.}
   \item{stanmodel}{The object of \code{stanmodel} that represents the Stan
     model and includes the compiled dynamic shared object for the model.}

  If the optimization is not finished for reasons such as feeding wrong data,
  a list with a component of the compiled model (an object of \code{stanmodel})
  is returned for future reusing. 
} 

\seealso{
  \code{\linkS4class{stanmodel}}
}

\examples{\dontrun{
m <- stan_model(model_code = 'parameters {real y;} model {y ~ normal(0,1);}')
f <- optimizing(m)
}} 

