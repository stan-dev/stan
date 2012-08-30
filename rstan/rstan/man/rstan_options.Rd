\name{rstan_options}
\alias{rstan_options}
\title{Set and read options used in RStan}
\description{
Set and read options used in RStan. Some settings as options can be 
controlled by the user.
}
\usage{
rstan_options(...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{Arguments of the form \code{opt = val} set
  option \code{opt} to value \code{val}.  Arguments of the
  form \code{opt} set the function to return option \code{opt}'s value.  
  Each argument must be a character string.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
  The available options are:
  \enumerate{
    \item \code{plot_rhat_breaks}: The cut off points for Rhat for which we would indicate
      using a different color. This is a numeric vector, defaults to \code{c(1.1, 1.2, 1.5, 2)}
      The value for this option will be sorted in ascending order. 
      So \code{plot_rhat_breaks = c(1.2, 1.5)} is equivalent to 
      \code{plot_rhat_breaks = c(1.5, 1.2)}. 
    \item \code{plot_rhat_cols}: A vector of the same length as   
          \code{plot_rhat_breaks} that indicates the colors for the
          breaks.
    \item \code{plot_rhat_na_col}: The color for Rhat when it is \code{NA}. 
    \item \code{plot_rhat_large_col}: The color for Rhat when it is larger than 
          the largest value of \code{plot_rhat_breaks}. 
    \item \code{rstan_alert_col}: The color used in method \code{plot}
          of S4 class \code{\linkS4class{stanfit}}  to show that the vector/array
          parameters are truncated. 
    \item \code{rstan_chains_cols}: The colors used in methods \code{plot}
          and \code{traceplot} of S4 class \code{\linkS4class{stanfit}}  
          for coloring different chains. 
    \item \code{rstawarmup_bg_col}: The background color for  
          the warmup area in the traceplots. 
    \item \code{boost_lib}: The path for the Boost C++ library used
          to compile Stan models.  This option is valid 
          for the whole \R session if not changed again. 
    \item \code{eigen_lib}: The path for the Eigen C++ library used
          to compile Stan models.  This option is valid 
         for the whole \R session if not changed again. 
  } 
}
\value{
The values as a \code{list} for existing options and \code{NA} for non-existent options.
When only one option is specified, its old value is returned. 
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
