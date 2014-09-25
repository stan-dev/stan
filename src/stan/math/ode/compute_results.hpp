#ifndef STAN__MATH__ODE__COMPUTE_RESULTS_HPP
#define STAN__MATH__ODE__COMPUTE_RESULTS_HPP

#include <vector>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>

namespace stan {
  namespace math {

    /**
     * add back initial states if y0 is var
     *
     * @param[in,out] y the states of the base system
     * @param[in] y0 the initial values for the base system
     */
    void add_initial_values(std::vector<std::vector<stan::agrad::var> >& y,
                            const std::vector<stan::agrad::var>& y0) {
      for (size_t n = 0; n < y.size(); n++)
        for (size_t m = 0; m < y0.size(); m++)
          y[n][m] += y0[m];
    }


    /**
     * Takes the coupled system and converts the result back to the
     * base ode system.
     *
     * This is the function for double initial values and double
     * theta. In this case, the coupled system is identical to the
     * base system.
     *
     * @param y the vector of the coupled states after solving the ode 
     * @param y0 initial state vector
     * @param theta parameter vector
     */
    std::vector<std::vector<double> > 
    compute_results(const std::vector<std::vector<double> >& y,
                    const std::vector<double>& y0,
                    const std::vector<double>& theta) {
      return y;
    }

    /**
     * Takes the coupled system and converts the result back to the
     * base ode system.
     *
     * This is the function for stan::agrad::var initial values and
     * double theta. If N is the number of states of the base ode and
     * M is the number of parameters (theta), this coupled system has
     * N * N states.
     *
     * @param y the vector of the coupled states after solving the ode 
     * @param y0 initial state vector
     * @param theta parameter vector
     */
    std::vector<std::vector<stan::agrad::var> > 
    compute_results(const std::vector<std::vector<double> >& y,
                    const std::vector<stan::agrad::var>& y0,
                    const std::vector<double>& theta) {

      std::vector<stan::agrad::var> temp_vars;
      std::vector<double> temp_gradients;
      std::vector<std::vector<stan::agrad::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        temp_vars.clear();
        
        //iterate over number of equations
        for (size_t j = 0; j < y0.size(); j++) { 
          temp_gradients.clear();
          
          //iterate over parameters for each equation
          for (size_t k = 0; k < y0.size(); k++)
            temp_gradients.push_back(y[i][y0.size() + y0.size()*k + j]);

          temp_vars.push_back(stan::agrad::precomputed_gradients(y[i][j], y0, temp_gradients));
        }

        y_return[i] = temp_vars;
      }

      add_initial_values(y_return, y0);

      return y_return;
    }

    /**
     * Takes the coupled system and converts the result back to the
     * base ode system.
     *
     * This is the function for double initial values and
     * double theta. If N is the number of states of the base ode and
     * M is the number of parameters (theta), this coupled system has
     * N * M states.
     *
     * @param y the vector of the coupled states after solving the ode 
     * @param y0 initial state vector
     * @param theta parameter vector
     */
    std::vector<std::vector<stan::agrad::var> > 
    compute_results(const std::vector<std::vector<double> >& y,
                    const std::vector<double>& y0,
                    const std::vector<stan::agrad::var>& theta) {

      std::vector<stan::agrad::var> temp_vars;
      std::vector<double> temp_gradients;
      std::vector<std::vector<stan::agrad::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        temp_vars.clear();
        
        //iterate over number of equations
        for (size_t j = 0; j < y0.size(); j++) { 
          temp_gradients.clear();
          
          //iterate over parameters for each equation
          for (size_t k = 0; k < theta.size(); k++)
            temp_gradients.push_back(y[i][y0.size() + y0.size()*k + j]);

          temp_vars.push_back(stan::agrad::precomputed_gradients(y[i][j], theta, temp_gradients));
        }

        y_return[i] = temp_vars;
      }

      return y_return;
    }

    /**
     * Takes the coupled system and converts the result back to the
     * base ode system.
     *
     * This is the function for double initial values and
     * double theta. If N is the number of states of the base ode and
     * M is the number of parameters (theta), this coupled system has
     * N * (N+M) states.
     *
     * @param y the vector of the coupled states after solving the ode 
     * @param y0 initial state vector
     * @param theta parameter vector
     */    
    std::vector<std::vector<stan::agrad::var> > 
    compute_results(const std::vector<std::vector<double> >& y,
                    const std::vector<stan::agrad::var>& y0,
                    const std::vector<stan::agrad::var>& theta) {
      std::vector<stan::agrad::var> vars = y0;
      vars.insert(vars.end(), theta.begin(), theta.end());

      std::vector<stan::agrad::var> temp_vars;
      std::vector<double> temp_gradients;
      std::vector<std::vector<stan::agrad::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        temp_vars.clear();
        
        //iterate over number of equations
        for (size_t j = 0; j < y0.size(); j++) { 
          temp_gradients.clear();
          
          //iterate over parameters for each equation
          for (size_t k = 0; k < y0.size()+theta.size(); k++)
            temp_gradients.push_back(y[i][y0.size() + y0.size()*k + j]);

          temp_vars.push_back(stan::agrad::precomputed_gradients(y[i][j], vars, temp_gradients));
        }

        y_return[i] = temp_vars;
      }

      add_initial_values(y_return, y0);

      return y_return;
    }

  }
}

#endif
