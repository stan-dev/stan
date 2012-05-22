

\name{stanc}
\alias{stanc} 
\docType{package}
\title{
Stanc: compile model in Stan language to C++ code
}
\description{
}
\usage{
stanc(model.code, model.name = "anon_model", verbose = FALSE) {
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
