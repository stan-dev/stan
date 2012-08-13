\name{stanc}
\alias{stanc} 
\docType{package}
\title{
\code{stanc}: compile model in Stan modeling language to C++ code
}
\description{
  Compile model in Stan language to C++ code, which could 
  be compiled and then loaded for drawing samples for the 
  model. 
}
\usage{
  stanc(model.code, model.name = "anon_model", verbose = FALSE) 
} 

\arguments{
  \item{model.code}{A character string for a model in Stan language.}
  \item{model.name}{A character string for naming the model. The 
    default is \code{anon_model}.}
  \item{verbose}{\code{TRUE} print out more intermediate information during the
    procedure; \code{FALSE} otherwise. The default is \code{FALSE}.}             
} 

\value{
  A list including an element of the C++ code. Relative elements are 
  \enumerate{ 
    \item \code{model.name} The model name specified by the user or the default if not specified. 
    \item \code{model.code} The model's Stan code. 
    \item \code{cppcode}    The model's C++ code. 
    \item \code{status}     The status indicating success/failure of compiling the Stan code. 
  } 
}

\note{
  R users might make the following mistakes when 
  using Stan modeling language.

  Using dot(".") in variable names such as \code{a.1}. However,
  The syntax in some respect of the Stan modeling language is
  more like programming language C++, which does not allow dot. 
  And in Stan modeling language, we need explicite semicolon (";")
  to end a statement. 

  In addition, all reserved key words in C++ programming language 
  could not be used as variable names, see Stan's manual. 
} 

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://code.google.com/p/stan}. 
} 

\seealso{
  \code{\link{stan.model}} and \code{\link{stan}}
} 

\examples{\dontrun{
stanmodelcode <- '
data {
  int[0,] N;
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
}}
