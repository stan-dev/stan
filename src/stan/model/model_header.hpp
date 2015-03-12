#ifndef STAN__MODEL__MODEL__HEADER_HPP__
#define STAN__MODEL__MODEL__HEADER_HPP__

#include <boost/exception/all.hpp>
#include <boost/random/linear_congruential.hpp>

// FIXME: this currently needs to be included first
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/meta/get.hpp>

#include <stan/math/prim/arr.hpp>
#include <stan/math/prim/mat.hpp>
#include <stan/math/prim/scal.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/arr.hpp>
#include <stan/math/rev/mat.hpp>
#include <stan/math/rev/scal.hpp>

#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/reader.hpp>
#include <stan/io/writer.hpp>
#include <stan/io/csv_writer.hpp>


#include <stan/model/prob_grad.hpp>
#include <stan/services/command.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#endif
