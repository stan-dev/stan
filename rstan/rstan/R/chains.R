# Part of the rstan package for an R interface to Stan 
# Copyright (C) 2012 Columbia University
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


rstan.ess <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  ess <- .Call("effective_sample_size", sim, n - 1, PACKAGE = "rstan")
  ess
} 

rstan.splitrhat <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  rhat <- .Call("split_potential_scale_reduction", sim, n - 1, PACKAGE = "rstan")
  rhat
} 
