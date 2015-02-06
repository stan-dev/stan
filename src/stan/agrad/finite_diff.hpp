#ifndef STAN__AGRAD__FINITE_DIFF_HPP
#define STAN__AGRAD__FINITE_DIFF_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/math/functions.hpp>
#include <stan/agrad/autodiff.hpp>

namespace stan {
  
  namespace agrad {

    template <typename F>
    double
    finite_diff_hess_helper(const F& f,
                            const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                            const int& lambda,
                            const double epsilon = 1e-03) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      Matrix<double,Dynamic,1> x_temp(x);

      double grad = 0.0;
      x_temp(lambda) = x(lambda) + 2.0 * epsilon;
      grad = -f(x_temp);

      x_temp(lambda) = x(lambda) + -2.0 * epsilon;
      grad += f(x_temp);

      x_temp(lambda) = x(lambda) + epsilon;
      grad += 8.0 * f(x_temp);

      x_temp(lambda) = x(lambda) + -epsilon;
      grad -= 8.0 * f(x_temp);

      return grad;
    }

    /**
     * Calculate the value and the gradient of the specified function
     * at the specified argument using finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double,Eigen::Dynamic,1>&)
     * </code>
     *
     * Error should be on order of epsilon ^ 6
     * 
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_fx Gradient of function at argument
     * @param[in] epsilon perturbation size
     */
    template <typename F>
    void
    finite_diff_gradient(const F& f,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                         double& fx,
                         Eigen::Matrix<double,Eigen::Dynamic,1>& grad_fx, 
                         const double epsilon = 1e-03) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      Matrix<double,Dynamic,1> x_temp(x);

      int d = x.size();
      grad_fx.resize(d);

      fx = f(x);
      
      for (int i = 0; i < d; ++i){
        double delta_f = 0.0;

        x_temp(i) += 3.0 * epsilon;
        delta_f = f(x_temp);

        x_temp(i) = x(i) + 2.0 * epsilon;
        delta_f -= 9.0 * f(x_temp);

        x_temp(i) = x(i) + epsilon;
        delta_f += 45.0 * f(x_temp);

        x_temp(i) = x(i) + -3.0 * epsilon;
        delta_f -= f(x_temp);

        x_temp(i) = x(i) + -2.0 * epsilon;
        delta_f += 9.0 * f(x_temp);

        x_temp(i) = x(i) + -epsilon;
        delta_f -= 45.0 * f(x_temp);

        delta_f /= 60 * epsilon;

        x_temp(i) = x(i);
        grad_fx(i) = delta_f;
      }

    }

    /**
     * Calculate the value and the Hessian of the specified function
     * at the specified argument using second-order finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double,Eigen::Dynamic,1>&)
     * </code>
     *
     * Error should be on order of epsilon ^ 4
     * 
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] hess_fx Hessian of function at argument
     * @param[in] epsilon perturbation size
     */

    template <typename F>
    void
    finite_diff_hessian(const F& f,
                        const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                        double& fx,
                        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& hess_fx, 
                        const double epsilon = 1e-03) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      int d = x.size();

      Matrix<double,Dynamic,1> x_temp(x);
      hess_fx.resize(d, d);

      fx = f(x);
      
      double f_diff(0.0);
      for (int i = 0; i < d; ++i){
        for (int j = i; j < d; ++j){
          x_temp(i) += 2.0 * epsilon;
          if(i != j){
            f_diff = -finite_diff_hess_helper(f,x_temp,j);
            x_temp(i) = x(i) + -2.0 * epsilon;
            f_diff += finite_diff_hess_helper(f,x_temp,j);
            x_temp(i) = x(i) + epsilon;
            f_diff += 8.0 * finite_diff_hess_helper(f,x_temp,j);
            x_temp(i) = x(i) + -epsilon;
            f_diff -= 8.0 * finite_diff_hess_helper(f,x_temp,j);
            f_diff /= 12.0 * epsilon * 12.0 * epsilon;
          } else {
            f_diff = -f(x_temp);
            f_diff -= 30 * fx;
            x_temp(i) = x(i) + -2.0 * epsilon;
            f_diff -= f(x_temp);
            x_temp(i) = x(i) + epsilon;
            f_diff += 16.0 * f(x_temp);
            x_temp(i) = x(i) - epsilon;
            f_diff += 16.0 * f(x_temp);
            f_diff /= 12 * epsilon * epsilon;
          }

          x_temp(i) = x(i);
          
          hess_fx(j,i) = f_diff;
          hess_fx(i,j) = hess_fx(j,i);
        }
      }
    }

    /**
     * Calculate the value and the hessian of the specified function
     * at the specified argument using first-order autodiff and 
     * first-order finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double,Eigen::Dynamic,1>&)
     * </code>
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] hess_fx Hessian of function at argument
     * @param[in] epsilon perturbation size
     */
    template <typename F>
    void
    finite_diff_hessian_auto(const F& f,
                             const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                             double& fx,
                             Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& hess_fx, 
                             const double epsilon = 1e-03) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using Eigen::VectorXd;

      int d = x.size();

      Matrix<double,Dynamic,1> x_temp(x);
      Matrix<double,Dynamic,1> g_auto(d);
      hess_fx.resize(d, d);

      fx = f(x);
      
      for (int i = 0; i < d; ++i){
        VectorXd g_diff = VectorXd::Zero(d);
        x_temp(i) += 2.0 * epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff = -g_auto;

        x_temp(i) = x(i) + -2.0 * epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff += g_auto;

        x_temp(i) = x(i) + epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff += 8.0 * g_auto;

        x_temp(i) = x(i) - epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff -= 8.0 * g_auto;

        x_temp(i) = x(i);
        g_diff /= 12.0 * epsilon;
        
        hess_fx.col(i) = g_diff;
      }

    }
    
    /** 
     * Calculate the value and the gradient of the hessian of the specified
     * function at the specified argument using second-order autodiff and
     * first-order finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * double
     * operator()(const
     * Eigen::Matrix<double,Eigen::Dynamic,1>&)
     * </code>
     *
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_hess_fx gradient of Hessian of function at argument
     * @param[in] epsilon perturbation size
     */
    template <typename F>
    void
    finite_diff_grad_hessian_auto(const F& f,
                                  const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                                  double& fx,
                                  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >& grad_hess_fx, 
                                  const double epsilon = 1e-04) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      int d = x.size();

      Matrix<double,Dynamic,1> x_temp(x);
      Matrix<double,Dynamic,1> grad_auto(d);
      Matrix<double,Dynamic,Dynamic> H_auto(d,d);
      Matrix<double,Dynamic,Dynamic> H_diff(d,d);

      fx = f(x);
      
      for (int i = 0; i < d; ++i){
        H_diff.setZero();

        x_temp(i) += 2.0 * epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff = -H_auto;

        x_temp(i) = x(i) + -2.0 * epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff += H_auto;

        x_temp(i) = x(i) + epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff += 8.0 * H_auto;

        x_temp(i) = x(i) + -epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff -= 8.0 * H_auto;

        x_temp(i) = x(i);
        H_diff /= 12.0 * epsilon;
        
        grad_hess_fx.push_back(H_diff);
      }

    }

  }
}
#endif
