#ifndef STAN__MATH__ODE__SOLVE_ODE_HPP__
#define STAN__MATH__ODE__SOLVE_ODE_HPP__

#include <ostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <stan/math/ode/util.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/operators/operator_addition.hpp>
#include <stan/agrad/rev/functions/fmax.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/error_handling/check_less.hpp>
#include <stan/math/error_handling/check_equal.hpp>
#include <stan/math/error_handling/matrix/check_matching_sizes.hpp>
#include <stan/math/error_handling/matrix/check_ordered.hpp>

namespace stan {
  namespace agrad {
     stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
       using std::fmax;
       return fmax(a, b);
     }
   }
 }

namespace stan {
  
  namespace math {

    // ODE coupled system for y0 double and theta double
    template <typename F, typename T1, typename T2>
    struct ode_system {
      const F& f_;
      const std::vector<double>& y0_;
      const std::vector<double>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      const int& num_eqn_;
      std::ostream* pstream_;
      ode_system(const F& f,
                 const std::vector<double>& y0,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const int& num_eqn,
                 std::ostream* pstream)
        : f_(f), 
          y0_(y0), 
          theta_(theta),
          x_(x), 
          x_int_(x_int), 
          num_eqn_(num_eqn),
          pstream_(pstream) { 
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double& t) {
        dy_dt = f_(t,y,theta_,x_,x_int_,pstream_);
        stan::math::check_matching_sizes("ode_system(%1%)",y,"y",dy_dt,"dy_dt",
                                         static_cast<double*>(0));
      }
    };

    // ODE coupled system for y0 double and theta var
    template <typename F>
    struct ode_system <F, double, stan::agrad::var> {
      const F& f_;
      const std::vector<double>& y0_;
      const std::vector<double>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      const int& num_eqn_;
      std::ostream* pstream_;
      ode_system(const F& f,
                 const std::vector<double>& y0,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const int& num_eqn,
                 std::ostream* pstream)
        : f_(f), 
          y0_(y0), 
          theta_(theta), 
          x_(x),
          x_int_(x_int), 
          num_eqn_(num_eqn),
          pstream_(pstream) { 
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double& t) {

        dy_dt = f_(t,y,theta_,x_,x_int_,pstream_);
        stan::math::check_equal("ode_system(%1%)",dy_dt.size(),num_eqn_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(num_eqn_ * theta_.size());

        std::vector<stan::agrad::var> theta_temp;
        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < num_eqn_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < num_eqn_; j++) {
            y_temp.push_back(y[j]);
            vars.push_back(y_temp[j]);
          }

          for (int j = 0; j < theta_.size(); j++) {
            theta_temp.push_back(theta_[j]);
            vars.push_back(theta_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);
          
          for (int j = 0; j < theta_.size(); j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[y_temp.size()+j];
            for (int k = 0; k < num_eqn_; k++)
              temp_deriv += y[num_eqn_+num_eqn_*j+k] * grad[k];

            coupled_sys[i+j*num_eqn_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }
    };
    
    // ODE coupled system for y0 var and theta double
    template <typename F>
    struct ode_system <F, stan::agrad::var, double> {
      const F& f_;
      const std::vector<double> y0_;
      const std::vector<double>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      const int& num_eqn_;
      std::ostream* pstream_;
      ode_system(const F& f,
                 const std::vector<double>& y0,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const int& num_eqn,
                 std::ostream* pstream)
        : f_(f), 
          y0_(y0), 
          theta_(theta), 
          x_(x), 
          x_int_(x_int), 
          num_eqn_(num_eqn),
          pstream_(pstream) { 
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double& t) {
        std::vector<double> y_new;
        for (int i = 0; i < num_eqn_; i++)
          y_new.push_back(y[i]+y0_[i]);
        dy_dt = f_(t,y_new,theta_,x_,x_int_,pstream_);
        stan::math::check_equal("ode_system(%1%)",dy_dt.size(),num_eqn_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(num_eqn_ * num_eqn_);

        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < num_eqn_; i++) {
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < num_eqn_; j++) {
            y_temp.push_back(y[j]+y0_[j]);
            vars.push_back(y_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);

          for (int j = 0; j < num_eqn_; j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[j];
            for (int k = 0; k < num_eqn_; k++)
              temp_deriv += y[num_eqn_+num_eqn_*j+k] * grad[k];

            coupled_sys[i+j*num_eqn_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }
    };

    // ODE coupled system for y0 var and theta var
    template <typename F>
    struct ode_system <F, stan::agrad::var, stan::agrad::var> {
      const F& f_;
      const std::vector<double> y0_;
      const std::vector<double>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      const int& num_eqn_;
      std::ostream* pstream_;
      ode_system(const F& f,
                 const std::vector<double> y0,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const int& num_eqn,
                 std::ostream* pstream)
        : f_(f), 
          y0_(y0), 
          theta_(theta), 
          x_(x), 
          x_int_(x_int), 
          num_eqn_(num_eqn),
          pstream_(pstream) { 
      }

      void operator()(const std::vector<double>& y,
                      std::vector<double>& dy_dt,
                      const double& t) {
        std::vector<double> y_new;
        for (int i = 0; i < num_eqn_; i++)
          y_new.push_back(y[i]+y0_[i]);
        dy_dt = f_(t,y_new,theta_,x_,x_int_,pstream_);
        stan::math::check_equal("ode_system(%1%)",dy_dt.size(),num_eqn_,"dy_dt",
                                static_cast<double*>(0));

        std::vector<double> coupled_sys(num_eqn_ * (num_eqn_+theta_.size()));

        std::vector<stan::agrad::var> theta_temp;
        std::vector<stan::agrad::var> y_temp;
        std::vector<stan::agrad::var> dy_dt_temp;
        std::vector<double> grad;
        std::vector<stan::agrad::var> vars;

        for (int i = 0; i < num_eqn_; i++) {
          theta_temp.clear();
          y_temp.clear();
          dy_dt_temp.clear();
          grad.clear();
          vars.clear();
          stan::agrad::start_nested();

          for (int j = 0; j < num_eqn_; j++) {
            y_temp.push_back(y[j]+y0_[j]);
            vars.push_back(y_temp[j]);
          }

          for (int j = 0; j < theta_.size(); j++) {
            theta_temp.push_back(theta_[j]);
            vars.push_back(theta_temp[j]);
          }

          dy_dt_temp = f_(t,y_temp,theta_temp,x_,x_int_,pstream_);
          dy_dt_temp[i].grad(vars, grad);

          for (int j = 0; j < num_eqn_+theta_.size(); j++) { 
            // orders derivatives by equation (i.e. if there are 2 eqns 
            // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as: 
            // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
            double temp_deriv = grad[j];
            for (int k = 0; k < num_eqn_; k++)
              temp_deriv += y[num_eqn_+num_eqn_*j+k] * grad[k];

            coupled_sys[i+j*num_eqn_] = temp_deriv;
          }

          stan::agrad::recover_memory_nested();
        }

        dy_dt.insert(dy_dt.end(), coupled_sys.begin(), coupled_sys.end());
      }
    };


    std::vector<std::vector<double> > 
    compute_results(const std::vector<std::vector<double> >& y,
                    const std::vector<double>& y0,
                    const std::vector<double>& theta) {
      return y;
    }


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

      return y_return;
    }

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

      return y_return;
    }

    template <typename F, typename T1, typename T2>
    std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
    integrate_ode(const F& f,
              const std::vector<T1> y0, 
              const double& t0, // initial time
              const std::vector<double>& ts, // times at desired solutions
              const std::vector<T2>& theta, // parameters
              const std::vector<double>& x, // double data values
              const std::vector<int>& x_int,
              std::ostream* pstream) { // int data values.
      using namespace boost::numeric::odeint;  // FIXME: trim to what is used
      stan::math::check_nonzero_size("integrate_ode(%1%)",ts,"time_vec",
                                     static_cast<double*>(0));
      stan::math::check_nonzero_size("integrate_ode(%1%)",y0,"y0_vec",
                                     static_cast<double*>(0));
      stan::math::check_ordered("integrate_ode(%1%)", ts, "times", 
                                static_cast<double*>(0));
      stan::math::check_less("integrate_ode(%1%)",t0,ts[0],"initial time",
                             static_cast<double*>(0));

      double absolute_tolerance = 1e-6;
      double relative_tolerance = 1e-6;

      std::vector<double> theta_dbl;
      for (int i = 0; i < theta.size(); i++)
        theta_dbl.push_back(value_of(theta[i]));

      std::vector<double> y0_vec; 
      for (size_t n = 0; n < y0.size(); n++)
        y0_vec.push_back(value_of(y0[n]));

      //initialize values to 0 if theta is var
      if (boost::is_same<stan::agrad::var, T2>::value)
        for (size_t n = 0; n < theta.size() * y0.size(); n++)
          y0_vec.push_back(0.0);

      //initalize values to 0 if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < y0.size() * y0.size(); n++)
          y0_vec.push_back(0.0);

      // builds coupled ode system
      ode_system<F, T1, T2> system(f,y0_vec,theta_dbl,x,x_int, y0.size(),pstream);

      // sets initial positions to 0 if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < y0.size(); n++)
          y0_vec[n] = 0;

      std::vector<double> ts_vec(ts.size()+1);
      ts_vec[0] = t0;
      for (size_t n = 0; n < ts.size(); n++)
        ts_vec[n+1] = ts[n];
      
      double step_size = 0.1;

      std::vector<std::vector<double> > x_vec;
      std::vector<double> t_vec;
      push_back_state_and_time<double> obs(x_vec, t_vec);
      integrate_times(
         make_dense_output(absolute_tolerance,
                           relative_tolerance,
                           runge_kutta_dopri5<std::vector<double>,
                                              double,std::vector<double>,double>()),
         system,
         y0_vec,
         boost::begin(ts_vec), boost::end(ts_vec), 
         step_size,
         obs);

      std::vector<std::vector<double> > y = obs.get();
      std::vector<std::vector<typename stan::return_type<T1,T2>::type> > 
        res = compute_results(y, y0, theta);

      res.erase(res.begin());

      // add back initial positions if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < res.size(); n++)
          for (size_t m = 0; m < y0.size(); m++)
            res[n][m] += y0[m];

      return res;
    }
                   
  }
}
#endif
