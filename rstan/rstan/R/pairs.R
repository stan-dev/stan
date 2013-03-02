# Part of the rstan package for statistical analysis via Markov Chain Monte Carlo
# Copyright (C) 2013 Stan Development Team
# Copyright (C) 1995-2012 The R Core Team
# Some parts  Copyright (C) 1999 Dr. Jens Oehlschlaegel-Akiyoshi
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

pairs.stanfit <-
  function (x, labels = NULL, panel = NULL, ..., 
            lower.panel = NULL, upper.panel = NULL, diag.panel = NULL, 
            text.panel = NULL, label.pos = 0.5 + has.diag/3, 
            cex.labels = NULL, font.labels = 1, row1attop = TRUE, gap = 1, 
            pars = NULL, chain_id = NULL) {
    
    if(is.null(pars)) pars <- dimnames(x)[[3]]
    arr <- extract(x, pars = pars, permuted = FALSE)
    sims <- nrow(arr)
    if(is.list(chain_id)) {
      if(length(chain_id) != 2) stop("if a list, 'chain_id' must be of length 2")
      arr <- arr[,c(chain_id[[1]], chain_id[[2]]),,drop = FALSE]
      k <- length(chain_id[[1]])
    }
    else if(!is.null(chain_id)) {
      arr <- arr[,chain_id,,drop = FALSE]
      k <- ncol(arr) %/% 2
    }
    else k <- ncol(arr) %/% 2
    mark <- 1:(sims * k)
    x <- apply(arr, MARGIN = "parameters", FUN = function(y) y)
    
    if(is.null(lower.panel)) {
      if(!is.null(panel)) lower.panel <- panel
      else lower.panel <- function(x,y, ...) {
        dots <- list(...)
        dots$x <- x[mark]
        dots$y <- y[mark]
        if(is.null(dots$nrpoints)) dots$nrpoints <- 0
        dots$add <- TRUE
        do.call(smoothScatter, args = dots)
      }
    }
    if(is.null(upper.panel)) {
      if(!is.null(panel)) upper.panel <- panel
      else upper.panel <- function(x,y, ...) {
        dots <- list(...)
        dots$x <- x[-mark]
        dots$y <- y[-mark]
        if(is.null(dots$nrpoints)) dots$nrpoints <- 0
        dots$add <- TRUE
        do.call(smoothScatter, args = dots)
      }
    }
    if(is.null(diag.panel)) diag.panel <- function(x, ...) {
        usr <- par("usr"); on.exit(par(usr))
        par(usr = c(usr[1:2], 0, 1.5) )
        h <- hist(x, plot = FALSE)
        breaks <- h$breaks; nB <- length(breaks)
        y <- h$counts; y <- y/max(y)
        rect(breaks[-nB], 0, breaks[-1], y, col="cyan", ...)
    }
    if(is.null(panel)) panel <- points
    
    if(is.null(text.panel)) textPanel <- function(x = 0.5, y = 0.5, txt, cex, font) {
      text(x,y, txt, cex = cex, font = font)
    }
    else textPanel <- text.panel
    has.diag <- TRUE
    if(is.null(labels)) labels <- colnames(x)

    mc <- match.call(expand.dots = FALSE)
    mc[1] <- call("pairs")
    mc$x <- x
    mc$labels <- labels
    mc$panel <- panel
    mc$lower.panel <- lower.panel
    mc$upper.panel <- upper.panel
    mc$diag.panel <- diag.panel
    mc$text.panel <- textPanel
    mc$chain_id <- NULL
    mc$pars <- NULL
    eval(mc)
  }
