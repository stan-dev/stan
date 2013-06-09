#ifndef __STAN__AGRAD__REV__MATRIX__ASSIGN_HPP__
#define __STAN__AGRAD__REV__MATRIX__ASSIGN_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_self.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace agrad {

    namespace {
      template <typename LHS, typename RHS>
      struct needs_promotion {
        enum { value = ( is_constant_struct<RHS>::value 
                         && !is_constant_struct<LHS>::value) };
      };
    
      template <bool PromoteRHS, typename LHS, typename RHS>
      struct assigner {
        static inline void assign(LHS& /*var*/, const RHS& /*val*/) {
          throw std::domain_error("should not call base class of assigner");
        }
      };
    
      template <typename LHS, typename RHS>
      struct assigner<false,LHS,RHS> {
        static inline void assign(LHS& var, const RHS& val) {
          var = val; // no promotion of RHS
        }
      };

      template <typename LHS, typename RHS>
      struct assigner<true,LHS,RHS> {
        static inline void assign(LHS& var, const RHS& val) {
          assign_to_var(var,val); // promote RHS
        }
      };
    }
    
    template <typename LHS, typename RHS>
    inline void assign(Eigen::Block<LHS> var, const RHS& val) {
      assigner<needs_promotion<Eigen::Block<LHS>,RHS>::value, Eigen::Block<LHS>, RHS>::assign(var,val);
    }
    
    template <typename LHS, typename RHS>
    inline void assign(LHS& var, const RHS& val) {
      assigner<needs_promotion<LHS,RHS>::value, LHS, RHS>::assign(var,val);
    }

    inline void assign(std::vector<double>& x, const std::vector<double>& y) {
      x = y;
    }

  }
}
#endif
