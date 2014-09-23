#ifndef STAN__MATH__ODE__ODE_SYSTEM_HPP
#define STAN__MATH__ODE__ODE_SYSTEM_HPP

#include <ostream>
#include <vector>
#include <stan/math/error_handling/check_equal.hpp>
#include <stan/math/error_handling/matrix/check_matching_sizes.hpp>

namespace stan {
  namespace math {

    // struct ode_system isn't broken out into a base class because it requires
    // this-> shenanigans and clunky constructor reuse everywhere
    template <typename F, typename T1, typename T2>
    struct ode_system {
    };

    
    // ODE coupled system for y0 double and theta double
    template <typename F>
    struct ode_system<F, double, double> {
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

  }
}

#endif
