#ifndef STAN_MATH_REV_CORE_VARI_HPP
#define STAN_MATH_REV_CORE_VARI_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <ostream>

namespace stan {
  namespace math {

    // forward declaration of var
    class var;

    /**
     * The variable implementation base class.
     *
     * A variable implementation is constructed with a constant
     * value.  It also stores the adjoint for storing the partial
     * derivative with respect to the root of the derivative tree.
     *
     * The chain() method applies the chain rule.  Concrete extensions
     * of this class will represent base variables or the result
     * of operations such as addition or subtraction.  These extended
     * classes will store operand variables and propagate derivative
     * information via an implementation of chain().
     */
    class vari : public chainable {
    private:
      friend class var;

    public:
      /**
       * The value of this variable.
       */
      const double val_;

      /**
       * The adjoint of this variable, which is the partial derivative
       * of this variable with respect to the root variable.
       */
      double adj_;

      /**
       * Construct a variable implementation from a value.  The
       * adjoint is initialized to zero.
       *
       * All constructed variables are added to the stack.  Variables
       * should be constructed before variables on which they depend
       * to insure proper partial derivative propagation.  During
       * derivative propagation, the chain() method of each variable
       * will be called in the reverse order of construction.
       *
       * @param x Value of the constructed variable.
       */
      explicit vari(const double x):
        val_(x),
        adj_(0.0) {
        ChainableStack::var_stack_.push_back(this);
      }

      vari(const double x, bool stacked):
        val_(x),
        adj_(0.0) {
        if (stacked)
          ChainableStack::var_stack_.push_back(this);
        else
          ChainableStack::var_nochain_stack_.push_back(this);
      }

      /**
       * Throw an illegal argument exception.
       *
       * <i>Warning</i>: Destructors should never called for var objects.
       *
       * @throw Logic exception always.
       */
      virtual ~vari() {
        // throw std::logic_error("vari destruction handled automatically");
      }

      /**
       * Initialize the adjoint for this (dependent) variable to 1.
       * This operation is applied to the dependent variable before
       * propagating derivatives.
       */
      virtual void init_dependent() {
        adj_ = 1.0;   // droot/droot = 1
      }

      /**
       * Set the adjoint value of this variable to 0.
       */
      virtual void set_zero_adjoint() {
        adj_ = 0.0;
      }

      /**
       * Insertion operator for vari. Prints the current value and
       * the adjoint value.
       *
       * @param os [in, out] ostream to modify
       * @param v [in] vari object to print.
       *
       * @return The modified ostream.
       */
      friend std::ostream& operator<<(std::ostream& os, const vari* v) {
        return os << v->val_ << ":" << v->adj_;
      }
    };

  }
}
#endif
