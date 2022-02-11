#ifndef STAN_MODEL_PROB_GRAD_HPP
#define STAN_MODEL_PROB_GRAD_HPP

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stan {

namespace model {

/**
 * Base class for models, holding the basic parameter sizes and
 * ranges for integer parameters.
 */
class prob_grad {
 protected:
  // TODO(carpenter): roll into model_base; remove int members/vars
  size_t num_params_r__;
  std::vector<std::pair<int, int> > param_ranges_i__;

 public:
  /**
   * Construct a model base class with the specified number of
   * unconstrained real parameters.
   *
   * @param num_params_r number of unconstrained real parameters
   */
  explicit prob_grad(size_t num_params_r)
      : num_params_r__(num_params_r),
        param_ranges_i__(std::vector<std::pair<int, int> >(0)) {}

  /**
   * Construct a model base class with the specified number of
   * unconstrained real parameters and integer parameter ranges.
   *
   * @param num_params_r number of unconstrained real parameters
   * @param param_ranges_i integer parameter ranges
   */
  prob_grad(size_t num_params_r,
            std::vector<std::pair<int, int> >& param_ranges_i)
      : num_params_r__(num_params_r), param_ranges_i__(param_ranges_i) {}

  /**
   * Destroy this class.
   */
  virtual ~prob_grad() {}

  /**
   * Return number of unconstrained real parameters.
   *
   * @return number of unconstrained real parameters
   */
  inline size_t num_params_r() const { return num_params_r__; }

  /**
   * Return number of integer parameters.
   *
   * @return number of integer parameters
   */
  inline size_t num_params_i() const { return param_ranges_i__.size(); }

  /**
   * Return the ordered parameter range for the specified integer
   * variable.
   *
   * @param idx index of integer variable
   * @throw std::out_of_range if there index is beyond the range
   * of integer indexes
   * @return ordered pair of ranges
   */
  inline std::pair<int, int> param_range_i(size_t idx) const {
    if (idx <= param_ranges_i__.size()) {
      std::stringstream ss;
      ss << "param_range_i(): No integer paramter at index " << idx;
      std::string msg = ss.str();
      throw std::out_of_range(msg);
    }
    return param_ranges_i__[idx];
  }
};
}  // namespace model
}  // namespace stan

#endif
