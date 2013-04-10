\name{stanc}
\alias{stanc} 
\docType{package}
\title{
Translate Stan model specification to C++ code
}
\description{
  Translate Stan model specification to C++ code, which can 
  then be compiled and loaded for sampling. 
}
\usage{
  stanc(file, model_code = '', model_name = "anon_model", verbose = FALSE, ...) 
} 

\arguments{
  \item{file}{A character string or a connection that \R supports 
    specifying the Stan model specification in Stan's modeling language.}
  \item{model_code}{A character string either containing a Stan model specification 
    or the name of a character string object in the workspace. 
    This parameter is used only if parameter \code{file} is 
    not specified, so it defaults to empty string.} 
  \item{model_name}{A character string naming the model. The 
    default is \code{"anon_model"}. However, the model name would be derived
    from \code{file} or \code{model_code} (if \code{model_code} is the name of a
    character string object) if \code{model_name} is not specified.} 
  \item{verbose}{\code{TRUE} print out more intermediate information during the
    translation procedure; \code{FALSE} otherwise. The default is \code{FALSE}.}             
  \item{\dots}{optional parameters including  
    \enumerate{ 
      \item \code{obfuscate_model_name} (logical), \code{TRUE} if not specified. 
      If \code{FALSE}, the model name in the generated C++ code would not 
      contain randomly generated character string so that if model names
      are given the same, the generated C++ code will have the same class
      name defining the model, and the same Rcpp module name, which is used 
      for R to execute the C++ code regardless if the stan models are the 
      same of not. Generally, it is recommented not to specify this parameter 
      or set it to \code{TRUE}. 
    }
  }
} 

\value{
  A list with named entries:
  \enumerate{ 
    \item \code{model_name} Character string for the model name.
    \item \code{model_code} Character string for the model's Stan specification. 
    \item \code{cppcode}    Character string for the model's C++ code. 
    \item \code{status}     Logical indicating success/failure (\code{TRUE/FALSE}) of 
                            translating the Stan code. 
  } 
}

\note{
  Unlike \R, in which variable identifiers may contain dots (e.g. \code{a.1}),
  Stan prohibits dots from occurring in variable identifiers.
  Further, C++ reserved words and
  Stan reserved words may not be used for variable names; see the
  Stan User's Guide for a complete list.  
} 

\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
} 

\seealso{
  \code{\link{stan_model}} and \code{\link{stan}}
} 

\examples{\dontrun{
stanmodelcode <- "
data {
  int<lower=0> N;
  real y[N];
} 

parameters {
  real mu;
} 

model {
  mu ~ normal(0, 10);
  y ~ normal(mu, 1); 
} 
"

r <- stanc(model_code = "stanmodelcode", model_name = "normal1") 
# r <- stanc(model_code = stanmodelcode, model_name = "normal1") 
names(r)
cat(r$cppcode)
}}
