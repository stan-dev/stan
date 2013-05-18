#ifndef __STAN__AGRAD__REV__MATRIX__ASSIGN_TO_VAR_HPP__
#define __STAN__AGRAD__REV__MATRIX__ASSIGN_TO_VAR_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/assign_to_var.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_self.hpp>

namespace stan {
  namespace agrad {

    // FIXME:  double val?
    inline void assign_to_var(stan::agrad::var& var, const double& val) {
      var = val;
    }
    inline void assign_to_var(stan::agrad::var& var, const stan::agrad::var& val) {
      var = val;
    }
    // FIXME:  int val?
    inline void assign_to_var(int& n_lhs, const int& n_rhs) {
      n_lhs = n_rhs;  // FIXME: no call -- just filler to instantiate
    }
    // FIXME:  double val?
    inline void assign_to_var(double& n_lhs, const double& n_rhs) {
      n_lhs = n_rhs;  // FIXME: no call -- just filler to instantiate
    }
    
    template <typename LHS, typename RHS>
    inline void assign_to_var(std::vector<LHS>& x, const std::vector<RHS>& y) {
      stan::math::validate_matching_sizes(x,y,"assign_to_var");
      for (size_t i = 0; i < x.size(); ++i)
        assign_to_var(x[i],y[i]);
    }
    template <typename LHS, typename RHS, int R, int C>
    inline void assign_to_var(Eigen::Matrix<LHS,R,C>& x, 
                              const Eigen::Matrix<RHS,R,C>& y) {
      stan::math::validate_matching_sizes(x,y,"assign_to_var");
      for (size_type n = 0; n < x.cols(); ++n)
        for (size_type m = 0; m < x.rows(); ++m)
          assign_to_var(x(m,n),y(m,n));
    }

    template <typename LHS, typename RHS, int R, int C>
    inline void assign_to_var(Eigen::Block<LHS>& x,
                              const Eigen::Matrix<RHS,R,C>& y) {
      stan::math::validate_matching_sizes(x,y,"assign_to_var");
      for (size_type n = 0; n < y.cols(); ++n)
        for (size_type m = 0; m < y.rows(); ++m)
          assign_to_var(x(m,n),y(m,n));
    }

  }
}
#endif
