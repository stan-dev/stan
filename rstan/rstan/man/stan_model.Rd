\name{stan_model}
\alias{stan_model}
\title{Construct a Stan model} 
\description{
  Construct an instance of S4 class \code{stanmodel} from a model 
  specified in Stan's modeling language. A \code{stanmodel} 
  can be used to draw samples from the model.  If the model is
  supplied in the Stan modeling language, it is first translated to
  C++ code.  The C++ code for the model plus other auxiliary 
  code is compiled into a dynamic shared object (DSO) and then loaded. 
  The loaded DSO for the model can be executed to draw samples, allowing
  inference to be performed for the model and data.
}

\usage{
  stan_model(file, model_name = "anon_model", 
             model_code = "", stanc_ret = NULL, 
             boost_lib = NULL, eigen_lib = NULL, 
             save_dso = TRUE, verbose = FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{file}{A character string or a connection that \R supports 
    specifying the Stan model specification in Stan's modeling language. 
  }

  \item{model_name}{A character string naming the model; defaults 
    to \code{"anon_model"}. However, the model name would be derived from 
    \code{file} or \code{model_code} (if \code{model_code} is the name of a
    character string object) if \code{model_name} is not specified.}

  \item{model_code}{A character string either containing the model 
    specification or the name of a character string object in the workspace;
    an alterantive is to specify the model with parameters \code{file}
    or \code{stanc_ret}. 
  }
  \item{stanc_ret}{A named list returned from a previous call to 
    function \code{stanc}. The list can be used to specify the model  
    instead of using parameter \code{file} or \code{model_code}. 
  }
  \item{boost_lib}{The path to a version of the Boost C++ library to
    use instead of the one supplied with RStan.}
  \item{eigen_lib}{The path to a version of the Eigen C++ library to
    use instead of the one in package \pkg{RcppEigen}.} 
  \item{save_dso}{Logical with the default of \code{TRUE}: indication of whether the 
    dynamic shared object (DSO) compiled from the C++ code for the model 
    will be saved or not. If \code{TRUE}, we can draw samples from
    the same model in another \R session using the saved DSO (i.e., without compiling 
    the C++ code again).}
  \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
    to report intermediate output to the console, 
    which might be helpful for debugging.}
  \item{\dots}{passed to \code{stanc}.}
}

\details{
  More details of Stan, including the full user's guide and reference 
  manual can be found at \url{http://mc-stan.org/}. 

  There are three ways to specify the model's code for \code{stan_model}. 
  \enumerate{ 
  \item parameter \code{model_code}, containing character string to 
       whose value is the Stan model specification,
  \item parameter \code{file}, indicating a file (or a connection) from
       which to read the Stan model specification, or 
  \item parameter \code{stanc_ret}, indicating the re-use of a model
       generated in a previous call to \code{stanc}.
  } 
}
\value{
  An instance of S4 class \code{\linkS4class{stanmodel}}, which can be
  used later for drawing samples by calling its \code{sampling} function.
}
\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
}
\seealso{
  \code{\linkS4class{stanmodel}} and \code{\link{stan}} 
}
\examples{\dontrun{
stan_model(model_code = 'parameters {real y;} model {y ~ normal(0,1);}')
}}
\keyword{rstan}
