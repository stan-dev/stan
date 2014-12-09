#ifndef STAN__AGRAD__REV__ODE__COUPLED_ODE_SYSTEM_HPP
#define STAN__AGRAD__REV__ODE__COUPLED_ODE_SYSTEM_HPP

#include <ostream>
#include <vector>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <stan/agrad/rev/operators/operator_plus_equal.hpp>
#include <stan/error_handling/scalar/check_equal.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>
#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace math {

    // This code is in this directory because it includes agrad::var
    // It is in namespace stan::math so that the partial template
    // specializations are treated as such.
    

    /**
     * Increment the state derived from the coupled system in the with
     * the original initial state.  This is necessary because the
     * coupled system subtracts out the initial state in its
     * representation when the initial state is unknown.  
     *
     * @param[in] y0 original initial values to add back into the
     * coupled system.
     * @param[in,out] y state of the coupled system on input,
     * incremented with initial values on output.
     */
    void add_initial_values(const std::vector<stan::agrad::var>& y0,
                            std::vector<std::vector<stan::agrad::var> >& y) {
      for (size_t n = 0; n < y.size(); n++)
        for (size_t m = 0; m < y0.size(); m++)
          y[n][m] += y0[m];
    }
    
    /**
     * The coupled ODE system for known initial values and unknown
     * parameters. 
     *
     * <p>If the base ODE state is size N and there are M parameters,
     * the coupled system has N + N * M states.
     * <p>The first N states correspond to the base system's N states:
     * \f$ \frac{d x_n}{dt} \f$
     * 
     * <p>The next M states correspond to the sensitivities of the 
     * parameters with respect to the first base system equation:
     * \f[ 
     *   \frac{d x_{N+m}}{dt}  
     *   = \frac{d}{dt} \frac{\partial x_1}{\partial \theta_m} 
     * \f]
     *
     * <p>The final M states correspond to the sensitivities with respect
     * to the second base system equation, etc.
     *
     * @tparam F type of functor for the base ode system.
     */
    template <typename F>
    struct coupled_ode_system <F, double, stan::agrad::var> {

      const F& f_;
      const std::vector<double>& y0_dbl_;
      const std::vector<stan::agrad::var>& theta_;
      std::vector<double> theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      const int N_;
      const int M_;
      const int size_;
      std::ostream* msgs_;

      /**
       * Construct a coupled ODE system with the specified base
       * ODE system, base initial state, parameters, data, and a
       * message stream.
       * 
       * @param[in] f the base ODE system functor.
       * @param[in] y0 the initial state of the base ode.
       * @param[in] theta parameters of the base ode.
       * @param[in] x real data.
       * @param[in] x_int integer data.
       * @param[in,out] msgs stream to which messages are printed.
       */
      coupled_ode_system(const F& f,
                         const std::vector<double>& y0,
                         const std::vector<stan::agrad::var>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* msgs)
        : f_(f), 
          y0_dbl_(y0),
          theta_(theta),
          theta_dbl_(theta.size(), 0.0),
          x_(x),
          x_int_(x_int), 
          N_(y0.size()),
          M_(theta.size()),
          size_(N_ + N_ * M_),
                msgs_(msgs) {

        for (int m = 0; m < M_; m++)
          theta_dbl_[m] = stan::agrad::value_of(theta[m]);
      }

      /**
       * Assign the derivative vector with the system derivatives at
       * the specified state and time.
       *
       * <p>The input state must be of size <code>size()</code>, and
       * the output produced will be of the same size.
       *
       * @param[in] y state of the coupled ode system.
       * @param[out] dy_dt populated with the derivatives of
       * the coupled system at the specified state and time.
       * @param[in]  t time.
       * @throw exception if the system function does not return the
       * same number of derivatives as the state vector size.
       */
      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      double t) {
        using std::vector;
        using stan::agrad::var;

        vector<double> y_base(y.begin(), y.begin()+N_);
        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,msgs_);
        stan::error_handling::check_equal("coupled_ode_system",
                                          "dy_dt", dy_dt.size(), N_);

        vector<double> coupled_sys(N_ * M_);
        vector<var> theta_temp;
        vector<var> y_temp;
        vector<var> dy_dt_temp;
        vector<double> grad;
        vector<var> vars;

        for (int i = 0; i < N_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          try {
            stan::agrad::start_nested();
            for (int j = 0; j < N_; j++) {
              y_temp.push_back(y[j]);
              vars.push_back(y_temp[j]);
            }

            for (int j = 0; j < M_; j++) {
              theta_temp.push_back(theta_dbl_[j]);
              vars.push_back(theta_temp[j]);
            }
            dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,msgs_);
            dy_dt_temp[i].grad(vars, grad);
          
            for (int j = 0; j < M_; j++) { 
              // orders derivatives by equation (i.e. if there are 2 eqns 
              // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
              // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
              double temp_deriv = grad[y_temp.size() + j];
              for (int k = 0; k < N_; k++)
                temp_deriv += y[N_ + N_ * j + k] * grad[k];

              coupled_sys[i + j * N_] = temp_deriv;
            }
          } catch (const std::exception& e) {
            stan::agrad::recover_memory_nested();
            throw;
          }
          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }
      
      /**
       * Returns the size of the coupled system.
       *
       * @return size of the coupled system.
       */
      int size() const {
        return size_;
      }

      /**
       * Returns the initial state of the coupled system.  Because the
       * initial values are known, the initial state of the coupled
       * system is the same as the initial state of the base ODE
       * system. 
       *
       * <p>This initial state returned is of size <code>size()</code>
       * where the first N (base ODE system size) parameters are the
       * initial conditions of the base ode system and the rest of the
       * initial condition elements are 0.
       *
       * @return the initial condition of the coupled system. 
       */
      std::vector<double> initial_state() {
        std::vector<double> state(size_, 0.0);
        for (int n = 0; n < N_; n++)
          state[n] = y0_dbl_[n];
        return state;
      }


      /**
       * Returns the base ODE system state corresponding to the
       * specified coupled system state. 
       *
       * @param y coupled states after solving the ode 
       */
      std::vector<std::vector<stan::agrad::var> > 
      decouple_states(const std::vector<std::vector<double> >& y) {
        using stan::agrad::precomputed_gradients;
        std::vector<stan::agrad::var> temp_vars;
        std::vector<double> temp_gradients;
        std::vector<std::vector<stan::agrad::var> > y_return(y.size());

        for (size_t i = 0; i < y.size(); i++) {
          temp_vars.clear();
        
          //iterate over number of equations
          for (size_t j = 0; j < N_; j++) { 
            temp_gradients.clear();
          
            //iterate over parameters for each equation
            for (size_t k = 0; k < M_; k++)
              temp_gradients.push_back(y[i][y0_dbl_.size() 
                                            + y0_dbl_.size() * k + j]);

            temp_vars.push_back(precomputed_gradients(y[i][j], 
                                                      theta_, 
                                                      temp_gradients));
          }
          y_return[i] = temp_vars;
        }

        return y_return;
      }

    };





    
    /**
     * The coupled ODE system for unknown initial values and known
     * parameters. 
     * 
     * <p>If the original ODE has states of size N, the 
     * coupled system has N + N * N states. (derivatives of each
     * state with respect to each initial value)
     *
     * <p>The coupled system has N + N * N states, where N is the size of
     * the state vector in the base system. 
     *
     * <p>The first N states correspond to the base system's N states:
     * \f$ \frac{d x_n}{dt} \f$
     *
     * <p>The next N states correspond to the sensitivities of the initial
     * conditions with respect to the to the first base system equation:
     * \f[ 
     *  \frac{d x_{N+n}}{dt}  
     *     = \frac{d}{dt} \frac{\partial x_1}{\partial y0_n}
     * \f]
     * 
     * <p>The next N states correspond to the sensitivities with respect
     * to the second base system equation, etc.
     *
     * @tparam F type of base ODE system functor
     */
    template <typename F>
    struct coupled_ode_system <F, stan::agrad::var, double> {

      const F& f_;
      const std::vector<stan::agrad::var>& y0_;
      std::vector<double> y0_dbl_;
      const std::vector<double>& theta_dbl_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* msgs_;
      const int N_;
      const int M_;
      const int size_;

      /**
       * Construct a coupled ODE system for an unknown initial state
       * and known parameters givne the specified base system functor,
       * base initial state, parameters, data, and an output stream
       * for messages.
       * 
       * @param[in] f base ODE system functor.
       * @param[in] y0 initial state of the base ODE.
       * @param[in] theta system parameters.
       * @param[in] x real data.
       * @param[in] x_int integer data.
       * @param[in,out] msgs output stream for messages.
       */
      coupled_ode_system(const F& f,
                         const std::vector<stan::agrad::var>& y0,
                         const std::vector<double>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* msgs)
        : f_(f), 
          y0_(y0),
          y0_dbl_(y0.size(), 0.0),
          theta_dbl_(theta), 
          x_(x), 
          x_int_(x_int), 
          msgs_(msgs),
          N_(y0.size()),
        M_(theta.size()),
        size_(N_ + N_ * N_) {

        for (int n = 0; n < N_; n++)
          y0_dbl_[n] = stan::agrad::value_of(y0_[n]);
      }

      /**
       * Calculates the derivative of the coupled ode system
       * with respect to the state y at time t.
       *
       * @param[in] y the current state of the coupled ode
       * system. This is a a vector of double of length size().
       * @param[out] dy_dt a vector of length size() with the
       * derivatives of the coupled system evaluated with state y and
       * time t.
       * @param[in] t time.
       * @throw exception if the system functor does not return a
       * derivative vector of the same size as the state vector.
       */
      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      double t) {
        std::vector<double> y_base(y.begin(), y.begin() + N_);
        for (int n = 0; n < N_; n++)
          y_base[n] += y0_dbl_[n];

        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,msgs_);
        stan::error_handling::check_equal("coupled_ode_system",
                                          "dy_dt", dy_dt.size(), N_);

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
          try {
            stan::agrad::start_nested();
            for (int j = 0; j < N_; j++) {
              y_temp.push_back(y[j] + y0_dbl_[j]);
              vars.push_back(y_temp[j]);
            }

            dy_dt_temp = f_(t,y_temp,theta_dbl_,x_,x_int_,msgs_);
            dy_dt_temp[i].grad(vars, grad);

            for (int j = 0; j < N_; j++) { 
              // orders derivatives by equation (i.e. if there are 2 eqns 
              // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
              // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
              double temp_deriv = grad[j];
              for (int k = 0; k < N_; k++)
                temp_deriv += y[N_ + N_ * j + k] * grad[k];

              coupled_sys[i+j*N_] = temp_deriv;
            }
          } catch (const std::exception& e) {
            stan::agrad::recover_memory_nested();
            throw;
          }
          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }

      /**
       * Returns the size of the coupled system.
       *
       * @return size of the coupled system.
       */
      int size() const {
        return size_;
      }

      /**
       * Returns the initial state of the coupled system.
       *
       * <p>Because the starting state is unknown, the coupled system
       * incorporates the initial conditions as parameters.  The
       * initial conditions for the coupled part of the system are set
       * to zero along with the rest of the initial state, because the
       * value of the initial state has been moved into the
       * parameters.
       *
       * @return the initial condition of the coupled system.
       *   This is a vector of length size() where all elements 
       *   are 0.
       */
      std::vector<double> initial_state() {
        return std::vector<double>(size_, 0.0);
      }

      /**
       * Return the solutions to the basic ODE system, including
       * appropriate autodiff partial derivatives, given the specified
       * coupled system solution.
       *
       * @param y the vector of the coupled states after solving the ode 
       */
      std::vector<std::vector<stan::agrad::var> > 
      decouple_states(const std::vector<std::vector<double> >& y) {
        using stan::agrad::precomputed_gradients;
        using stan::agrad::var;
        using std::vector;

        vector<var> temp_vars;
        vector<double> temp_gradients;
        vector<vector<var> > y_return(y.size());

        for (size_t i = 0; i < y.size(); i++) {
          temp_vars.clear();
        
          // iterate over number of equations
          for (size_t j = 0; j < N_; j++) { 
            temp_gradients.clear();
          
            // iterate over parameters for each equation
            for (size_t k = 0; k < N_; k++)
              temp_gradients.push_back(y[i][y0_.size() + y0_.size() * k + j]);

            temp_vars.push_back(precomputed_gradients(y[i][j], 
                                                      y0_, temp_gradients));
          }

          y_return[i] = temp_vars;
        }

        add_initial_values(y0_, y_return);

        return y_return;
      }

    };






    
    /**
     * The coupled ode system for unknown intial values and unknown
     * parameters. 
     *
     * <p>The coupled system has N + N * (N + M) states, where N is
     * size of the base ODE state vector and M is the number of
     * parameters.
     *
     * <p>The first N states correspond to the base system's N states:
     *   \f$ \frac{d x_n}{dt} \f$
     * 
     * <p>The next N+M states correspond to the sensitivities of the
     * initial conditions, then to the parameters with respect to the
     * to the first base system equation:
     *
     * \f[ 
     *   \frac{d x_{N + n}}{dt}
     *     = \frac{d}{dt} \frac{\partial x_1}{\partial y0_n}
     * \f]
     *
     * \f[ 
     *   \frac{d x_{N+N+m}}{dt}
     *     = \frac{d}{dt} \frac{\partial x_1}{\partial \theta_m}
     * \f]
     * 
     * <p>The next N+M states correspond to the sensitivities with
     * respect to the second base system equation, etc.
     *
     * <p>If the original ode has a state vector of size N states and
     * a parameter vector of size M, the coupled system has N + N * (N
     * + M) states. (derivatives of each state with respect to each
     * initial value and each theta)
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
      const int N_;
      const int M_;
      const int size_;
      std::ostream* msgs_;

      /**
       * Construct a coupled ODE system with unknown initial value and
       * known parameters, given the base ODE system functor, the
       * initial state of the base ODE, the parameters, data, and an
       * output stream to which to write messages.
       * 
       * @param[in] f the base ode system functor.
       * @param[in] y0 the initial state of the base ode.
       * @param[in] theta parameters of the base ode.
       * @param[in] x real data.
       * @param[in] x_int integer data.
       * @param[in,out] msgs output stream to which to print messages.
       */
      coupled_ode_system(const F& f,
                         const std::vector<stan::agrad::var>& y0,
                         const std::vector<stan::agrad::var>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int,
                         std::ostream* msgs)
        : f_(f), 
          y0_(y0),
          y0_dbl_(y0.size(), 0.0),
          theta_(theta), 
          theta_dbl_(theta.size(), 0.0), 
          x_(x), 
          x_int_(x_int), 
          N_(y0.size()),
          M_(theta.size()),          
          size_(N_ + N_ * (N_ + M_)),
          msgs_(msgs) {

        for (int n = 0; n < N_; n++)
          y0_dbl_[n] = stan::agrad::value_of(y0[n]);

        for (int m = 0; m < M_; m++)
          theta_dbl_[m] = stan::agrad::value_of(theta[m]);
      }

      /**
       * Populates the derivative vector with derivatives of the
       * coupled ODE system state with respect to time evaluated at the
       * specified state and specified time.
       *
       * @param[in]  y the current state of the coupled ode system,
       * of size <code>size()</code>.
       * @param[in,out] dy_dt populate with the derivatives of the
       * coupled system evaluated at the specified state and time.
       * @param[in] t time.
       * @throw exception if the base system does not return a
       * derivative vector of the same size as the state vector.
       */
      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      double t) {
        using std::vector;
        using stan::agrad::var;
          
        vector<double> y_base(y.begin(), y.begin()+N_);
        for (int n = 0; n < N_; n++)
          y_base[n] += y0_dbl_[n];

        dy_dt = f_(t,y_base,theta_dbl_,x_,x_int_,msgs_);
        stan::error_handling::check_equal("coupled_ode_system",
                                          "dy_dt", dy_dt.size(), N_);

        vector<double> coupled_sys(N_ * (N_ + M_));
        vector<var> theta_temp;
        vector<var> y_temp;
        vector<var> dy_dt_temp;
        vector<double> grad;
        vector<var> vars;

        for (int i = 0; i < N_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          try {
            stan::agrad::start_nested();

            for (int j = 0; j < N_; j++) {
              y_temp.push_back(y[j] + y0_dbl_[j]);
              vars.push_back(y_temp[j]);
            }

            for (int j = 0; j < M_; j++) {
              theta_temp.push_back(theta_dbl_[j]);
              vars.push_back(theta_temp[j]);
            }

            dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,msgs_);
            dy_dt_temp[i].grad(vars, grad);

            for (int j = 0; j < N_+M_; j++) { 
              // orders derivatives by equation (i.e. if there are 2 eqns 
              // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
              // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
              double temp_deriv = grad[j];
              for (int k = 0; k < N_; k++)
                temp_deriv += y[N_ + N_ * j + k] * grad[k];

              coupled_sys[i + j * N_] = temp_deriv;
            }
          } catch (const std::exception& e) {
            stan::agrad::recover_memory_nested();
            throw;
          }
          stan::agrad::recover_memory_nested();
        }
        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }

      /**
       * Returns the size of the coupled system.
       *
       * @return size of the coupled system.
       */
      int size() const {
        return size_;
      }

      /**
       * Returns the initial state of the coupled system.  
       *
       * Because the initial state is unknown, the coupled system
       * incorporates the initial condition offset from zero as
       * a parameter, and hence the return of this function is a
       * vector of zeros.
       *
       * @return the initial condition of the coupled system.  This is
       * a vector of length size() where all elements are 0.
       */
      std::vector<double> initial_state() {
        return std::vector<double>(size_, 0.0);
      }

      /**
       * Return the basic ODE solutions given the specified coupled
       * system solutions, including the partials versus the
       * parameters encoded in the autodiff results.
       *
       * @param y the vector of the coupled states after solving the ode 
       */    
      std::vector<std::vector<stan::agrad::var> > 
      decouple_states(const std::vector<std::vector<double> >& y) {
        using std::vector;
        using stan::agrad::var;
        using stan::agrad::precomputed_gradients;

        vector<var> vars = y0_;
        vars.insert(vars.end(), theta_.begin(), theta_.end());
        
        vector<var> temp_vars;
        vector<double> temp_gradients;
        vector<vector<var> > y_return(y.size());
        
        for (size_t i = 0; i < y.size(); i++) {
          temp_vars.clear();
          
          //iterate over number of equations
          for (size_t j = 0; j < N_; j++) { 
            temp_gradients.clear();
            
            //iterate over parameters for each equation
            for (size_t k = 0; k < N_ + M_; k++)
              temp_gradients.push_back(y[i][N_ + N_ * k + j]);

            temp_vars.push_back(precomputed_gradients(y[i][j], 
                                                      vars, temp_gradients));
          }
          y_return[i] = temp_vars;
        }
        add_initial_values(y0_, y_return);
        return y_return;
      }

    };


  }

}

#endif
