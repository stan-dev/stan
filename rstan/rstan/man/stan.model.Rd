\name{stan.model}
\alias{stan.model}
\title{Construct a Stan model} 
\description{
  Construct and \code{stanmodel} object from a model 
  represented by Stan modeling language. The model 
  is compiled to C++ code if it is not done.  
  The model C++ code plus other auxiliary 
  code is compiled into shared object and then loaded. 
  The loaded module is available for drawing samples.  
}

\usage{
  stan.model(file, model.name = "anon_model", 
             model.code = "", stanc.ret = NULL, boost.lib = NULL, 
             save.dso = TRUE, verbose = FALSE)
}
%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{file}{A character string or a connection that R supports specifying the model 
    in Stan language. 
  }

  \item{model.name}{A character string for naming the model (the
    default is \code{"anon_model"}). 
  }

  \item{model.code}{A character string to specify the model instead of using
    argument file. 
  }
  \item{stanc.ret}{The returned list from \code{stanc}, an alternative way
    to specify the model using previous results.}   
  \item{boost.lib}{The path for boost library \url{http://http://www.boost.org/users/download/}. 
    If specified, this path would be used instead for boost library header files for compiling 
    the model in C++ language.}
  \item{save.dso}{Logical with the default of \code{TRUE}: indication of whether the 
    dynamic shared object (DSO) compiled from the C++ code for the model 
    will be saved or not. If \code{TRUE}, we can draw samples from
    the same model in another R session using the saved object of class \code{stanfit} (i.e.,
    returned from function \code{stan}) without compiling the C++ code again.}
  \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
    intermediate output inside the function, which might be helpful for debugging
    the model, are printed or not. 
  }
}

\details{
  %%  ~~ If necessary, more details than the description above ~~
  The details of Stan can be found at \url{http://code.googlecode.com/p/stan}. 

  There are three ways to specify the model's code for \code{stan.model}. 
  \enumerate{ 
  \item use a character string to include the Stan modeling language code 
        and then specify parameter \code{model.code}. 
  \item use a file and then specify file parameter \code{file}. 
  \item use the compiled results from \code{stanc} and then specify 
        parameter \code{stanc.ret}. 
  } 
}
\value{
  A stanmodel class object (\linkS4class{stanmodel}), which can be
  used later for drawing samples by calling its \code{sampling} function.
}
\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://code.google.com/p/stan}. 
}
\note{
  The object of S4 class \code{stanmodel} created by \code{stan.model} 
  is well saved when calling function \code{stan} (or \code{stan.model})
  with \code{save.dso = TRUE}. 
}

\seealso{
  \code{\linkS4class{stanmodel}} and \code{\link{stan}} 
}
\examples{\dontrun{
stan.model(model.code = 'parameters {real y;} model {y ~ normal(0,1);}')
}}
\keyword{rstan}
