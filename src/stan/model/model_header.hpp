#ifndef STAN_MODEL_MODEL_HEADER_HPP
#define STAN_MODEL_MODEL_HEADER_HPP

#include <stan/model/model_base.hpp>
#include <stan/model/model_base_crtp.hpp>
#include <stan/math.hpp>

#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/io/opencl/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#endif

#include <stan/io/deserializer.hpp>
#include <stan/io/serializer.hpp>

#include <stan/model/rethrow_located.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/model/indexing.hpp>
#include <stan/services/util/create_rng.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#endif
