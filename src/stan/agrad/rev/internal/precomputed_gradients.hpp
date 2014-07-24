#ifndef STAN__AGRAD__REV__INTERNAL__PRECOMPUTED_GRADIENTS_HPP
#define STAN__AGRAD__REV__INTERNAL__PRECOMPUTED_GRADIENTS_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {
    
    /**
     * This is a var implementation class that
     * takes precomputed gradient values.
     *
     * Stan users should use function precomputed_gradients
     * directly.
     */
    class precomputed_gradients_vari : public vari {
    protected:
      std::vector<vari *> varis_;
      std::vector<double> gradients_;

    public:
      /**
       * Constructs a precomputed_gradients_vari.
       * The the value, pointers to the varis, and
       * gradient values need to be provided.
       * 
       * Note: this class is slow since it
       *
       * @param val The value of the variable.
       * @param varis Vector of pointers to the varis of 
       *   independent variables of this class.
       * @param gradients Vector of gradients with respect 
       *   to the independent variables. These must be indexed
       *   in the same order as varis.
       * @throws std::invalid_argument if the sizes don't match
       */
      precomputed_gradients_vari(const double val,
                                 std::vector<vari *>& varis,
                                 const std::vector<double>& gradients) 
        : vari(val),
          varis_(varis),
          gradients_(gradients) {
        if (varis_.size() != gradients_.size())
          throw std::invalid_argument("sizes of varis and gradients do not match");
      }
      
      /**
       * Implements the chain rule for this variable.
       *
       * Each of the independent variables' adjoints
       * are updated with the appropriate gradient
       * value.
       */
      void chain() {
        for (size_t n = 0; n < varis_.size(); n++) {
          varis_[n]->adj_ += adj_ * gradients_[n];
        }
      }
    };

    
    /**
     * This function is provided for Stan users 
     * that want to compute gradients without
     * using Stan's auto-diff.
     *
     * Users need to provide the value, the independent
     * variables, and the gradients of this expression with
     * respect to the indepedent variables.
     *
     * (For advanced users, a faster version that 
     *  doesn't involve copying vectors exists can
     *  be written.)
     *
     * @param value The value of the resulting dependent variable.
     * @param vars The independent variables.
     * @param gradients The value of the gradients of the dependent 
     *   variable with respect to the independent variables.
     * @returns An auto-diff variable that uses the precomputed 
     *   gradients provided.
     */
    var precomputed_gradients(const double value,
                              const std::vector<var>& vars,
                              const std::vector<double>& gradients) {
      std::vector<vari *> varis;
      varis.resize(vars.size());
      for (size_t n = 0; n < vars.size(); n++) {
        varis[n] = vars[n].vi_;
      }
      return var(new precomputed_gradients_vari(value, varis, gradients));
    }
  }
}
#endif
