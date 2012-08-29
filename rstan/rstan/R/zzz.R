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

.onLoad <- function(libname, pkgname) {
    # require("methods", character = TRUE, quietly = TRUE)
    # loadRcppModules()
}

.onAttach <- function(...) {
  rstanLib <- dirname(system.file(package = "rstan"))
  version <- packageDescription("rstan", lib.loc = rstanLib)$Version
  builddate <- packageDescription("rstan", lib.loc = rstanLib)$Packaged
  gitrev <- substring(git_head(), 0, 12) 
  packageStartupMessage(paste("rstan (Version ", version, ", packaged: ", builddate, ", GitRev: ", gitrev, ")", sep = ""))
} 

