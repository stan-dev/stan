#ifndef STAN__AGRAD__REV__JACOBIAN_HPP
#define STAN__AGRAD__REV__JACOBIAN_HPP

// ********* here because it's only used for testing **********
// ********* superseded by version in autodiff.hpp for API ****

#include <vector>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the Jacobian of the function producing the specified
     * dependent variables with respect to the specified independent
     * variables. 
     *
     * A typical use case would be to take the Jacobian of a function
     * from independent variables to dependentant variables.  For instance,
     * 
     * <pre>
     * std::vector<var> f(std::vector<var>& x) { ... }
     * std::vector<var> x = ...;
     * std::vector<var> y = f(x);
     * std::vector<std::vector<double> > J;
     * jacobian(y,x,J);
     * </pre>
     *
     * After executing this code, <code>J</code> will contain the
     * Jacobian, stored as a standard vector of gradients.
     * Specifically, <code>J[m]</code> will be the gradient of <code>y[m]</code>
     * with respect to <code>x</code>, and thus <code>J[m][n]</code> will be 
     * <code><i>d</i>y[m]/<i>d</i>x[n]</code>.
     *
     * @param[in] dependents Dependent (output) variables.
     * @param[in] independents Indepent (input) variables.
     * @param[out] jacobian Jacobian of the transform.
     */
    inline void jacobian(std::vector<var>& dependents,
                         std::vector<var>& independents,
                         std::vector<std::vector<double> >& jacobian) {
      jacobian.resize(dependents.size());
      for (size_t i = 0; i < dependents.size(); ++i) {
        jacobian[i].resize(independents.size());
        if (i > 0) 
          set_zero_all_adjoints();
        jacobian.push_back(std::vector<double>(0));
        grad(dependents[i].vi_);
        for (size_t j = 0; j < independents.size(); ++j)
          jacobian[i][j] = independents[j].adj();
      }
    }

  }
}
#endif
