#ifndef STAN__MATH__PRIM__ARR__FUNCTOR__COUPLED_ODE_OBSERVER_HPP
#define STAN__MATH__PRIM__ARR__FUNCTOR__COUPLED_ODE_OBSERVER_HPP

#include <vector>

namespace stan {

  namespace math {

    /**
     * Observer for the coupled states.  Holds a reference to
     * an externally defined vector of vectors passed in at
     * construction time.
     */
    struct coupled_ode_observer {

      std::vector<std::vector<double> >& y_coupled_;
      int n_;

      /**
       * Construct a coupled ODE observer from the specified coupled
       * vector.
       *
       * @param y_coupled reference to a vector of vector of doubles.
       */
      coupled_ode_observer(std::vector<std::vector<double> >& y_coupled)
        : y_coupled_(y_coupled), n_(0) {
      }

      /**
       * Callback function for Boost's ODE solver to record values.
       *
       * @param coupled_state solution at the specified time.
       * @param t time of solution.
       */
      void operator()(const std::vector<double>& coupled_state,
                      const double t) {
        y_coupled_[n_] = coupled_state;
        n_++;
      }

    };

  }

}

#endif
