\name{stan}
\alias{stan}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit a Stan model
%%  ~~function to do ... ~~
}
\description{
Fit a model in Stan model language and return 
fitted result. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
stan(file, model.name = "anon_model", model.code = "", data, n.chains = 1L,
n.iter = 2000L, n.warmup = floor(n.iter/2), n.thin = 1L, init.t = "random",
init.v = NULL, seed, sample.file, stan.home, verbose = FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{file}{
%%     ~~Describe \code{file} here~~
}
  \item{model.name}{
%%     ~~Describe \code{model.name} here~~
}
  \item{model.code}{
%%     ~~Describe \code{model.code} here~~
}
  \item{data}{
%%     ~~Describe \code{data} here~~
}
  \item{n.chains}{
%%     ~~Describe \code{n.chains} here~~
}
  \item{n.iter}{
%%     ~~Describe \code{n.iter} here~~
}
  \item{n.warmup}{
%%     ~~Describe \code{n.warmup} here~~
}
  \item{n.thin}{
%%     ~~Describe \code{n.thin} here~~
}
  \item{init.t}{
%%     ~~Describe \code{init.t} here~~
}
  \item{init.v}{
%%     ~~Describe \code{init.v} here~~
}
  \item{seed}{
%%     ~~Describe \code{seed} here~~
}
  \item{sample.file}{
%%     ~~Describe \code{sample.file} here~~
}
  \item{stan.home}{
%%     ~~Describe \code{stan.home} here~~
}
  \item{verbose}{
%%     ~~Describe \code{verbose} here~~
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{

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

  s.f <- stan(model.code = scode, verbose = FALSE) 
  print(s.f)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{rstan} 
