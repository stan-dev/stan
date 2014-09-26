#ifndef STAN__MATH__ODE__COUPLED_ODE_SYSTEM_HPP
#define STAN__MATH__ODE__COUPLED_ODE_SYSTEM_HPP

#include <ostream>
#include <vector>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <stan/agrad/rev/operators/operator_plus_equal.hpp>
#include <stan/math/error_handling/check_equal.hpp>
#include <stan/math/error_handling/matrix/check_matching_sizes.hpp>
#include <stan/meta/traits.hpp>

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
     * Structure for the coupled ordinary differential equation
     * system.  The coupled ode system is the a new system consisting
     * of the base system coupled with the sensitivities.
     * 
     * Implementation notes:
     * - struct coupled_ode_system isn't broken out into a base class because 
     *   it requires this-> shenanigans and clunky constructor reuse 
     *   everywhere
     * - The default templated structure does nothing. This will prevent
     *   the T1 and T2 types from being instantiated as something other
     *   than double and stan::agrad::var.
     * 
     * @tparam F the functor for the base ode system
     * @tparam T1 type of the initial state
     * @tparam T2 type of the parameters
     */
    template <typename F, typename T1, typename T2>
    struct coupled_ode_system {
    };

    
    /**
     * The coupled ode system for double initial values and 
     * double theta. This coupled system is identical to the
     * the base system.
     *
     * @tparam F the functor for the base ode system
     */
    template <typename F>
    struct coupled_ode_system<F, double, double> {
      const F& f_;
      const std::vector<double>& y0_dbl_;
      const std::vector<double>& theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* pstream_;
      const int N_;
      const int M_;
      const int size_;

      coupled_ode_system(const F& f,
                         const std::vector<double>& y0,
                         const std::vector<double>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* pstream)
        : f_(f), 
          y0_dbl_(y0),
          theta_dbl_(theta),
          x_(x), 
          x_int_(x_int), 
          pstream_(pstream),
          N_(y0.size()),
          M_(theta.size()),
          size_(N_) {
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double t) {
        dy_dt = f_(t,y,theta_dbl_,x_,x_int_,pstream_);
        stan::math::check_matching_sizes("coupled_ode_system(%1%)",y,"y",dy_dt,"dy_dt",
                                         static_cast<double*>(0));
      }

      const int size() const {
        return size_;
      }

      std::vector<double> initial_state() {
        std::vector<double> state(size_, 0.0);
        // initial values to y0 because we don't need the
        // sensitivies of y0.
        for (int n = 0; n < N_; n++)
          state[n] = y0_dbl_[n];
        return state;
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
      
    };

    /**
     * The coupled ode system for double initial values and 
     * stan::agrad::var theta. If the original ode has N
     * states and M thetas, the new coupled system has
     * N + N * M states.
     *
     *
     *
     * @tparam F the functor for the base ode system
     */
    template <typename F>
    struct coupled_ode_system <F, double, stan::agrad::var> {
      const F& f_;
      const std::vector<double>& y0_;
      const std::vector<stan::agrad::var>& theta_;
      std::vector<double> theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* pstream_;
      const int N_;
      const int M_;
      const int size_;

      coupled_ode_system(const F& f,
                         const std::vector<double>& y0,
                         const std::vector<stan::agrad::var>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* pstream)
        : f_(f), 
          y0_(y0),
          theta_(theta),
          theta_dbl_(theta.size(), 0.0),
          x_(x),
          x_int_(x_int), 
          pstream_(pstream),
          N_(y0.size()),
        M_(theta.size()),
        size_(N_ + N_ * M_) {
        // setup theta
        for (int m = 0; m < M_; m++)
          theta_dbl_[m] = value_of(theta[m]);
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double t) {
        std::vector<double> y_base(y.begin(), y.begin()+N_);
        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,pstream_);
        stan::math::check_equal("coupled_ode_system(%1%)",dy_dt.size(),N_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(N_ * M_);

        std::vector<stan::agrad::var> theta_temp;
        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < N_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < N_; j++) {
            y_temp.push_back(y[j]);
            vars.push_back(y_temp[j]);
          }

          for (int j = 0; j < M_; j++) {
            theta_temp.push_back(theta_dbl_[j]);
            vars.push_back(theta_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);
          
          for (int j = 0; j < M_; j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[y_temp.size()+j];
            for (int k = 0; k < N_; k++)
              temp_deriv += y[N_+N_*j+k] * grad[k];

            coupled_sys[i+j*N_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }
      
      const int size() const {
        return size_;
      }

      std::vector<double> initial_state() {
        std::vector<double> state(size_, 0.0);
        // initial values to y0 because we don't need the
        // sensitivies of y0.
        for (int n = 0; n < N_; n++)
          state[n] = y0_[n];
        return state;
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

    };

    
    /**
     * The coupled ode system for stan::agrad::var initial values and
     * double theta. If the original ode has N states and M thetas,
     * the new coupled system has N + N * N states. (derivatives
     * of each state with respect to each initial value)
     *
     * @tparam F the functor for the base ode system
     */
    template <typename F>
    struct coupled_ode_system <F, stan::agrad::var, double> {
      const F& f_;
      const std::vector<stan::agrad::var>& y0_;
      std::vector<double> y0_dbl_;
      const std::vector<double>& theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* pstream_;
      const int N_;
      const int M_;
      const int size_;

      coupled_ode_system(const F& f,
                         const std::vector<stan::agrad::var>& y0,
                         const std::vector<double>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* pstream)
        : f_(f), 
          y0_(y0),
          y0_dbl_(y0.size(), 0.0),
          theta_dbl_(theta), 
          x_(x), 
          x_int_(x_int), 
          pstream_(pstream),
          N_(y0.size()),
        M_(theta.size()),
        size_(N_ + N_ * N_) {

        for (int n = 0; n < N_; n++)
          y0_dbl_[n] = value_of(y0_[n]);
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double t) {
        std::vector<double> y_base(y.begin(), y.begin()+N_);
        for (int n = 0; n < N_; n++)
          y_base[n] += y0_dbl_[n];

        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,pstream_);
        stan::math::check_equal("coupled_ode_system(%1%)",dy_dt.size(),N_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(N_ * N_);

        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < N_; i++) {
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < N_; j++) {
            y_temp.push_back(y[j]+y0_dbl_[j]);
            vars.push_back(y_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_dbl_,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);

          for (int j = 0; j < N_; j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[j];
            for (int k = 0; k < N_; k++)
              temp_deriv += y[N_+N_*j+k] * grad[k];

            coupled_sys[i+j*N_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }

      const int size() const {
        return size_;
      }


      std::vector<double> initial_state() {
        std::vector<double> state(size_, 0.0);
        // initial values are all 0 because we're getting sensitivities
        // of y0
        return state;
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

    };

    
    /**
     * The coupled ode system for stan::agrad::var initial values and
     * stan::agrad::var theta. If the original ode has N states and M thetas,
     * the new coupled system has N + N * (N + M) states. (derivatives
     * of each state with respect to each initial value and each theta)
     *
     * @tparam F the functor for the base ode system
     */
    template <typename F>
    struct coupled_ode_system <F, stan::agrad::var, stan::agrad::var> {
      const F& f_;
      const std::vector<stan::agrad::var>& y0_;
      std::vector<double> y0_dbl_;
      const std::vector<stan::agrad::var>& theta_;
      std::vector<double> theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* pstream_;
      const int N_;
      const int M_;
      const int size_;

      coupled_ode_system(const F& f,
                         const std::vector<stan::agrad::var>& y0,
                         const std::vector<stan::agrad::var>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* pstream)
        : f_(f), 
          y0_(y0),
          y0_dbl_(y0.size(), 0.0),
          theta_(theta), 
          theta_dbl_(theta.size(), 0.0), 
          x_(x), 
          x_int_(x_int), 
        pstream_(pstream),
        N_(y0.size()),
        M_(theta.size()),          
        size_(N_ + N_ * (N_ + M_)) {
        // setup y0
        for (int n = 0; n < N_; n++)
          y0_dbl_[n] = value_of(y0[n]);

        // setup theta
        for (int m = 0; m < M_; m++)
          theta_dbl_[m] = value_of(theta[m]);

      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double t) {
        std::vector<double> y_base(y.begin(), y.begin()+N_);
        for (int n = 0; n < N_; n++)
          y_base[n] += y0_dbl_[n];

        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,pstream_);
        stan::math::check_equal("coupled_ode_system(%1%)",dy_dt.size(),N_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(N_ * (N_+M_));
        
        std::vector<stan::agrad::var> theta_temp;
        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < N_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < N_; j++) {
            y_temp.push_back(y[j]+y0_dbl_[j]);
            vars.push_back(y_temp[j]);
          }

          for (int j = 0; j < M_; j++) {
            theta_temp.push_back(theta_dbl_[j]);
            vars.push_back(theta_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);

          for (int j = 0; j < N_+M_; j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[j];
            for (int k = 0; k < N_; k++)
              temp_deriv += y[N_+N_*j+k] * grad[k];

            coupled_sys[i+j*N_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }

      const int size() const {
        return size_;
      }

      std::vector<double> initial_state() {
        std::vector<double> state(size_, 0.0);
        // initial values are all 0 because we're getting sensitivities
        // of y0
        return state;
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

    };

  }
}

#endif
