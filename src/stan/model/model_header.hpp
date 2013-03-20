#ifndef STAN__MODEL__MODEL__HEADER_HPP__
#define STAN__MODEL__MODEL__HEADER_HPP__

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/exception/all.hpp>

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/agrad/matrix_error_handling.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/gm/command.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/reader.hpp>
#include <stan/io/writer.hpp>
#include <stan/io/csv_writer.hpp>
#include <stan/math.hpp>
#include <stan/math/matrix.hpp>

// FIXME: these should go in matrix.hpp
#include <stan/math/matrix/add.hpp>
#include <stan/math/matrix/cholesky_decompose.hpp>
#include <stan/math/matrix/col.hpp>
#include <stan/math/matrix/cumulative_sum.hpp>
#include <stan/math/matrix/diag_matrix.hpp>
#include <stan/math/matrix/divide.hpp>
#include <stan/math/matrix/eigenvalues_sym.hpp>
#include <stan/math/matrix/elt_divide.hpp>
#include <stan/math/matrix/elt_multiply.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/math/matrix/mdivide_left.hpp>
#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <stan/math/matrix/mdivide_right.hpp>
#include <stan/math/matrix/minus.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/matrix/row.hpp>
#include <stan/math/matrix/singular_values.hpp>
#include <stan/math/matrix/softmax.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/transpose.hpp>

#include <stan/mcmc/sampler.hpp>
#include <stan/model/prob_grad_ad.hpp>
#include <stan/prob/distributions.hpp>

#endif
