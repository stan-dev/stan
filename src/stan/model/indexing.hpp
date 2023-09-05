#ifndef STAN_MODEL_INDEXING_HPP
#define STAN_MODEL_INDEXING_HPP

#include <stan/model/indexing/access_helpers.hpp>
#ifdef STAN_OPENCL
#include <stan/model/indexing/assign_cl.hpp>
#include <stan/model/indexing/rvalue_cl.hpp>
#endif
#include <stan/model/indexing/assign_varmat.hpp>
#include <stan/model/indexing/assign.hpp>
#include <stan/model/indexing/deep_copy.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/rvalue_varmat.hpp>
#include <stan/model/indexing/rvalue.hpp>

#endif
