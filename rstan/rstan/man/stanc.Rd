\name{stanc}
\alias{stanc} 
\docType{package}
\title{
\code{stanc}: translate Stan model specification to C++ code
}
\description{
  Translate Stan model specification to C++ code, which can 
  then be compiled and loaded for sampling. 
}
\usage{
  stanc(model_code, model_name = "anon_model", verbose = FALSE) 
} 

\arguments{
  \item{model_code}{A character string containing a Stan model specification.}
  \item{model_name}{A character string naming the model. The 
    default is \code{anon_model}.}
  \item{verbose}{\code{TRUE} print out more intermediate information during the
    translation procedure; \code{FALSE} otherwise. The default is \code{FALSE}.}             
} 

\value{
  A list with named entries:
  \enumerate{ 
    \item \code{model_name} Character string for the model name.
    \item \code{model_code} Character string for the model's Stan specification. 
    \item \code{cppcode}    Character string for the model's C++ code. 
    \item \code{status}     Boolean status indicating success/failure of 
                            translating and compiling the Stan code. 
  } 
}

\note{
  Unlike \R, in which variable identifiers may contain dots (e.g. \code{a.1}),
  Stan prohibits dots from occurring in variable identifiers.
  using Stan modeling language.  Further, C++ reserved words and
  Stan reserved words may not be used for variable names;  see the
  Stan User's Guide for a complete list.  
} 

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference Manual}. 
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

r <- stanc(stanmodelcode, "normal1") 
names(r)
cat(r$cppcode)
}}
