
\name{pairs.stanfit}
\alias{pairs.stanfit}

\title{Create a matrix of output plots from a \code{stanfit} object} 

\description{A \code{\link{pairs}} method that is customized for MCMC output} 

\usage{
  \method{pairs}{stanfit}(x, labels = NULL, panel = NULL, \dots, lower.panel = NULL, 
    upper.panel = NULL, diag.panel = NULL, text.panel = NULL, 
    label.pos = 0.5 + has.diag/3, cex.labels = NULL, font.labels = 1, 
    row1attop = TRUE, gap = 1, pars = NULL, chain_id = NULL) 
} 

\arguments{
  \item{x}{An object of S4 class \code{stanfit}}
  \item{labels, panel, \dots, lower.panel, upper.panel, diag.panel}{Same as in 
    \code{\link{pairs}} syntactically but see the Details section for different
    default arguments}
  \item{label.pos, cex.labels, font.labels, row1attop, gap}{Same as in 
    \code{\link{pairs}}}
  \item{pars}{If not \code{NULL}, a character vector indicating which quantities
    to include in the plots, which is passed to \code{\link{extract}}. Thus,
    by default, all unknown quantities are included, which may be far too many
    to visualize on a small computer screen.}
  \item{chain_id}{By default, \code{NULL}, which will plot roughly half of the
    chains in the lower panel and the rest in the upper panel. Otherwise, an
    integer vector can be passed to select some subset of the chains, of which
    roughly half will be plotted in the lower panel and the rest in the upper
    panel. Or a list of two integer vectors, each specifying a subset of the
    chains to be plotted in the lower and upper panels respectively. The latter
    option can be useful if you suspect that one chain has failed to converge.}  
} 

\details{
  This method differs from the default \code{\link{pairs}} method in the following
  ways. If unspecified, the \code{\link{smoothScatter}} function is used for the
  off-diagonal plots, rather than \code{\link{points}}, since the former is more
  appropriate for visualizing thousands of draws from a posterior distribution.
  Also, if unspecified, histograms of the marginal distribution of each quantity
  are placed on the diagonal of the plot, after pooling all of the chains specified
  by the \code{chain\_id} argument.
  
  The draws from the warmup phase are always discarded before plotting.
}
  
\seealso{
  S4 class \code{\linkS4class{stanfit}} and its method \code{\link{extract}} as
  well as the \code{\link{pairs}} generic function
} 

\examples{\dontrun{
ex_model_code <- '
  parameters {
    real alpha[2,3];
    real beta[2]; 
  } 
  model {
    for (i in 1:2) for (j in 1:3) 
      alpha[i, j] ~ normal(0, 1); 
    for (i in 1:2) 
      beta[i] ~ normal(0, 2); 
    # beta ~ normal(0, 2) // vectorized version
  } 
'

## fit the model 
fit <- stan(model_code = ex_model_code, chains = 4) 

pairs(fit, pars = "alpha", chain_id = list(1, 2:4))
}}  
