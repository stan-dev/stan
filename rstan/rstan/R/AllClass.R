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

setClass(Class = "cxxdso",
         representation = representation(
           sig = "list", # A list of function signature that would be returned by cxxfuncion 
           dso_saved = "logical", # flag for if the dso is saved or not
           # dso_last_path = 'character', # where the dso is saved last time 
           dso_filename = "character", # the dso file name when it is created the first time
           system = "character", # what is the OS (R.version$system)?  
           .MISC = "environment" # an envir to save 
                                 #  1. the returned object by cxxfuncion using name cxxfun 
                                 #  2. the file path used last time using name dso_last_path 
                                 #  3. The binary dso with name dso.bin, which is a raw vector.  
                                 #     We put it here since the environment is not copied 
                                 #     when assigned to another 
                                 #     http://cran.r-project.org/doc/manuals/R-lang.html#Environment-objects
         ),
         validity = function(object) {
           length(object@sig) > 0 && identical(object@system, R.version$system)
         })


setClass(Class = "stanfit",
         representation = representation(
           model_name = "character", 
           model_pars = "character", 
           par_dims = "list", 
           sim = "list", 
           inits = "list", 
           stan_args = "list", 
           .MISC = "environment"
         ),  
         validity = function(object) {
           return(TRUE) 
         })

setClass(Class = "stanmodel",
         representation = representation(
           model_name = "character",
           model_code = "character",
           .modelmod = "environment", 
             # to store the sampler object, Rcpp module stanfit instance 
             # and the model's name in C++ code (model_cppname), also
             # used for the Rcpp module name. 
           dso = 'cxxdso'), 
         validity = function(object) {
           return(TRUE)
         })

# S4 class cxxdso is defined at cxxdso-class.R
