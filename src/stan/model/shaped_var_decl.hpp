#ifndef STAN_MODEL_SHAPED_VAR_DECL_HPP
#define STAN_MODEL_SHAPED_VAR_DECL_HPP

#include <stan/model/sized_var_decl.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace model {

    /**
     * Runtime variable declaration information including the name of
     * the variable, the name of its type, number of array dimensions,
     * the sizes of all of the dimensions (including vector and
     * matrix dimensions, with array dimensions first), and lower and
     * upper bounds (if any).
     */
    class shaped_var_decl : public sized_var_decl {
    private:
      const double lower_bound_;
      const double upper_bound_;

    public:
      /**
       * Construct a variable declaration with the specified name and
       * type name, array dimensions, lower and upper bound indicator
       * flags, dimension sizes, and lower and upper bounds.
       *
       * @param[in] name name of variable
       * @param[in] type_name name of the variable's type
       * @param[in] array_dims number of array dimensions
       * @param[in] has_lb true if declaration has lower
       * bound
       * @param[in] has_ub true if declaration has upper
       * bound
       * @param[in] sizes sizes of dimensions, including array and
       * matrix dimensions
       * @param[in] lb lower bound (or NaN)
       * @param[in] ub upper bound (or NaN)
       */
      shaped_var_decl(const std::string& name, const std::string& type_name,
                      int array_dims, bool has_lb, bool has_ub,
                      const std::vector<int>& sizes, double lb, double ub)
        : sized_var_decl(name, type_name, array_dims, has_lb, has_ub, sizes),
          lower_bound_(lb), upper_bound_(ub) { }

      /**
       * Return the lower bound constraint or NaN if there is no lower
       * bound.
       *
       * @return lower bound
       */
      double lower_bound() const {
        return lower_bound_;
      }

      /**
       * Return the upper bound constraint or NaN if there is no upper
       * bound.
       *
       * @return upper bound
       */
      double upper_bound() const {
        return upper_bound_;
      }
    };

  }
}
#endif
