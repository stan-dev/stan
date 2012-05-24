
\name{rstan-package}
\alias{rstan-package}
\alias{rstan} 
\docType{package}
\title{RStan --- R interface to Stan}
\description{
  R interface to Stan (see \url{http:/stan.googlecode.com}). 
}

\details{
  \tabular{ll}{
    Package: \tab rstan\cr
    Type: \tab Package\cr
    Version: \tab 1.0\cr
    Date: \tab 2012-05-14\cr
    License: \tab GPL-3 \cr
  }

  The RStan package provides an interface to Stan. 
  See the manual of Stan for more information. 

}
\author{
  Who wrote it

  Maintainer: Who to complain to <yourfault@somewhere.net>
}
\references{
 \url{http://stan.googlecode.com} 
}


\keyword{stan, rstan, HMC, NUTS} 
\seealso{
  % ~~ Optional links to other man pages, e.g. ~~
  % ~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
  %% ~~ simple examples of the most important functions ~~
  cat("Stan version: ", stanc.version(), sep = ''); 

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
  rr <- stan.model(model.code = stanmodelcode, model.name = "normal1", 
                   verbose = TRUE) 

  y <- rnorm(20) 
  dat <- list(N = 20, y = y); 
  samples(rr, data = dat, n.iter = 2012, sample.file = 'norm1.csv')
} 
