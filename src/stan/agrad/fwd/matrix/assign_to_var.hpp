#ifndef __STAN__AGRAD__FWD__MATRIX__ASSIGN_TO_VAR_HPP__
#define __STAN__AGRAD__FWD__MATRIX__ASSIGN_TO_VAR_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    template<typename T>
    inline void assign_to_var(stan::agrad::fvar<T>& fvar, const double& val) {
      fvar = val;
    }

    template<typename T>
    inline void assign_to_var(stan::agrad::fvar<T>& fvar, const int& val) {
      fvar = val;
    }
  }
}
#endif
