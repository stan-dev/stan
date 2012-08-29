\name{rstan_options}
\alias{rstan_options}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Set and read options used in RStan
%%  ~~function to do ... ~~
}
\description{
Set and read options used in RStan. Some arguments are programmed as options
so that they can be controlled by user. 
}
\usage{
rstan_options(...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{Arguments must be in forms of \code{argument = value}.
  Then option \code{argument} is set as \code{value}. In the case
  of just obtaining options, each argument can be a character string.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
  Currently, there are following options:
  \enumerate{
    \item \code{plot_rhat_breaks}: The cut off points for Rhat for which we would indicate
      using a different color. This is a numeric vector, defaults to \code{c(1.1, 1.2, 1.5, 2)}. 
    \item \code{plot_rhat_cols}: The vector of colors for indicating Rhat, which are 
      corresponding to \code{plot_rhat_breaks}. So it should have the same 
      length of \code{plot_rhat_breaks}. 
    \item \code{plot_rhat_na_col}: The color for Rhat when it is \code{NA}. 
    \item \code{plot_rhat_large_col}: The color for Rhat when it is larger than 
      all in \code{plot_rhat_breaks}. 
    \item \code{rstan_alert_col}: The color in \code{plot(signature = 'stanfit')} 
      to show that the vector/array parameters are truncated. 
    \item \code{rstan_chains_cols}: The colors used in \code{plot(signature = 'stanfit')}
      and \code{traceplot(signature = 'stanfit')} for coloring different chains. 
    \item \code{rstawarmup_bg_col}: The background color for the warmup area in the traceplots. 
    \item \code{boost_lib}: The path for the boost library. The user can specify 
     \code{boost_lib} for calling \code{stan}. This option is valid 
     for the whole \R session if not changed again. 
  } 
}
\value{
A list of the old options if exists. If an option does
not exist before, it is \code{NA}. When there is only 
one option, the old value is returned. 
}
%\references{
%% ~put references to the literature/web site here ~
%}
%\author{
%%  ~~who you are~~
%}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
%\examples{}
\keyword{rstan} 
