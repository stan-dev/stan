

\name{stanc}
\alias{stanc} 
\docType{package}
\title{
Stanc: compile model in Stan language to C++ code
}
\description{
  Compile model in Stan language to C++ code, which could later 
  be compiled and loaded for drawing samples from the 
  model. 
}
\usage{
  stanc(model.code, model.name = "anon_model", verbose = FALSE) 
} 

\arguments{
  \item{model.code}{A character string for a model in Stan language.}
  \item{model.name}{A character string for nameing the model. The 
    default is \code{"anon_model"}.}
  \item{verbose}{\code{TRUE} provide more information during the procedure; 
    \code{FALSE} otherwise. The default is \code{FALSE}.}             
} 

\examples{
stanmodelcode <- '
data {
  int(0,) N;
  real y[N];
} 

parameters {
  real mu;
} 

model {
  mu ~ normal(0, 10);
  y ~ normal(mu, 1); 
} 

'
r <- stanc(stanmodelcode, "normal1") 
names(r)
cat(r$cppcode)
\dontrun{
}
}
