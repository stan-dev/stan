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
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

setClass(Class = "stanfit",
         representation = representation(
           model.name = "character", 
           model.pars = "character", 
           par.dims = "list", 
           sim = "list", 
           inits = "list", 
           stan.args = "list", 
           .MISC = "environment"
         ),  
         validity = function(object) {
           return(TRUE) 
         })

setClass(Class = "stanmodel",
         representation = representation(
           model.name = "character",
           model.code = "character",
           .modelmod = "list"
         ), 
         validity = function(object) {
           return(TRUE)
         })
