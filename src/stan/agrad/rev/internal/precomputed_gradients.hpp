#ifndef STAN__AGRAD__REV__INTERNAL__PRECOMPUTED_GRADIENTS_HPP
#define STAN__AGRAD__REV__INTERNAL__PRECOMPUTED_GRADIENTS_HPP

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {

  namespace agrad {
    
    /**
     * A variable implementation taking a sequence of operands and
     * partial derivatives with respect to the operands.
     *
     * Stan users should use function precomputed_gradients()
     * directly.
     */
    class precomputed_gradients_vari : public vari {
    protected:

      const size_t size_;
      vari** varis_;
      double* gradients_;

    public:

      /**
       * Construct a precomputed vari with the specified value,
       * operands, and gradients.
       * 
       * @param[in] val The value of the variable.
       * @param[in] vars Vector of operands.
       * @param[in] gradients Vector of partial derivatives of value
       * with respect to operands.
       * @throws std::invalid_argument if the sizes of the vectors
       * don't match.
       */
      precomputed_gradients_vari(const double val,
                                 const std::vector<var>& vars,
                                 const std::vector<double>& gradients)
        : vari(val),
          size_(vars.size()),
          varis_(ChainableStack::memalloc_
                 .alloc_array<vari*>(vars.size())),
          gradients_(ChainableStack::memalloc_
                     .alloc_array<double>(vars.size())) {
        if (vars.size() != gradients.size())
          throw std::invalid_argument("sizes of vars and gradients"
                                      " do not match");
        for (size_t i = 0; i < vars.size(); ++i)
          varis_[i] = vars[i].vi_; 
        std::copy(gradients.begin(), gradients.end(), gradients_);
      }
      
      /**
       * Implements the chain rule for this variable, using the
       * prestored operands and gradient. 
       */
      void chain() {
        for (size_t i = 0; i < size_; ++i) 
          varis_[i]->adj_ += adj_ * gradients_[i];
      }
    };


    /**
     * This function returns a var for an expression that has the
     * specified value, vector of operands, and vector of partial
     * derivatives of value with respect to the operands.
     *
     * @param[in] value The value of the resulting dependent variable.
     * @param[in] operands operands.
     * @param[in] gradients vector of partial derivatives of result with
     * respect to operands.
     * @return An auto-diff variable that uses the precomputed 
     *   gradients provided.
     */
    var precomputed_gradients(const double value,
                              const std::vector<var>& operands,
                              const std::vector<double>& gradients) {
      return var(new precomputed_gradients_vari(value, operands, gradients));
    }
  }
}
#endif
