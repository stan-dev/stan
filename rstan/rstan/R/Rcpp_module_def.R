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


get_Rcpp_module_def_code <- function(model.name) {
  def_Rcpp_module_hpp_file <- 
    system.file('include', '/rstan/rcpp_module_def_for_rstan.hpp', package = 'rstan') 
  if (def_Rcpp_module_hpp_file == '') 
    stop("Rcpp module definition file for rstan is not found.\n") 
  src <- paste(readLines(def_Rcpp_module_hpp_file), collapse = '\n')
  gsub("%model_name%", model.name, src)
} 

