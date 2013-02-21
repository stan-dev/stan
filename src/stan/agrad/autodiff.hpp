#ifndef __STAN__AGRAD__AUTO_DIFF_HPP__
#define __STAN__AGRAD__AUTO_DIFF_HPP__

#define EIGEN_DENSEBASE_PLUGIN "stan/math/EigenDenseBaseAddons.hpp"
#include <Eigen/Dense>

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/fvar.hpp>

namespace stan {
  
  namespace agrad {

    /**
     * Calculate the value and the gradient of the specified function
     * at the specified argument.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * stan::agrad::var
     * operator()(const
     * Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>&)
     * </code>
     *
     * using only operations that are defined for
     * <code>stan::agrad::var</code>.  This latter constraint usually
     * requires the functions to be defined in terms of the libraries
     * defined in Stan or in terms of functions with appropriately
     * general namespace imports that eventually depend on functions
     * defined in Stan.
     *
     * <p>Time and memory usage is on the order of the size of the
     * fully unfolded expression for the function applied to the
     * argument, independently of dimension.
     * 
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_fx Gradient of function at argument
     */
    template <typename F>
    void
    gradient(const F& f,
             const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
             double& fx,
             Eigen::Matrix<double,Eigen::Dynamic,1>& grad_fx) {
      using stan::agrad::var;
      Eigen::Matrix<var,Eigen::Dynamic,1> x_var(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_var(i) = x(i);
      var fx_var = f(x_var);
      fx = fx_var.val();

      grad_fx.resize(x.size());
      stan::agrad::grad(fx_var.vi_);
      for (int i = 0; i < x.size(); ++i)
        grad_fx(i) = x_var(i).adj();
      recover_memory();
    }

    template <typename F, typename T>
    void
    directional_derivative(const F& f,
                           const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                           const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
                           T& fx,
                           T& dfx_dv) {
      using stan::agrad::fvar;
      using stan::agrad::var;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      Matrix<fvar<T>,Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_fvar(i) = fvar<T>(x(i),v(i));
      fvar<T> fx_fvar = f(x);
      fx = fx_fvar.val_;
      dfx_dv = fx.d_;
    }
                           

    template <typename F>
    void
    hessian_times_vector(const F& f,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& v,
                         double& fx,
                         Eigen::Matrix<double,Eigen::Dynamic,1>& Hv) {
      using stan::agrad::fvar;
      using stan::agrad::var;
      using Eigen::Matrix; using Eigen::Dynamic;
      Matrix<var,Dynamic,1> x_var(x.size());
      for (int i = 0; i < x_var.size(); ++i)
        x_var(i) = x(i);
      Matrix<fvar<var>,Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_fvar[i] = fvar<var>(x_var(i), v(i));  
      fvar<var> fx_fvar = f(x_fvar);
      fx = fx_fvar.val_.val();
      stan::agrad::grad(fx_fvar.d_.vi_);
      Hv.resize(x.size());
      for (int i = 0; i < x.size(); ++i) 
        Hv(i) = x_var(i).adj();
    }


    // // N * bk
    // template <typename F>
    // void
    // jacobian_fast(const F& f,
    //               const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
    //               Eigen::Matrix<double,Eigen::Dynamic,1>& fx,
    //               Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamci>& J);


    // // 1 * fwd
    // template <typename F, typename S>
    // void
    // partial(const F& f,
    //         const Eigen::Matrix<S,Eigen::Dynamic,1>& x,
    //         S& fx,
    //         int i,
    //         S& dfx_dxi); 

    // // 1 * fwd
    // template <typename F, typename S>
    // void
    // vector_dot_gradient(const F& f,
    //                     const Eigen::Matrix<S,Eigen::Dynamic,1> x,
    //                     S& fx,
    //                     Eigen::Matrix<S,Eigen::Dynamic,1> v,
    //                     S& v_dot_grad_fx);

        
    // // N * N * fwd
    // template <typename F, typename S>
    // void
    // jacobian(const F& f,
    //          const Eigen::Matrix<S,Eigen::Dynamic,1>& x,
    //          Eigen::Matrix<S,Eigen::Dynamic,1>& fx,
    //          Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>& J);

    // // N * (fwd + rev)
    // template <typename F>
    // void
    // hessian_fast(const F& f,
    //              const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
    //              double& fx,
    //              const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& H);

    // // N * N * fwd(2nd order)
    // template <typename F, typename S>
    // void
    // hessian(const F& f,
    //         const Eigen::Matrix<S,Eigen::Dynamic,1>& x,
    //         S& fx,
    //         const Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>& H);


    // // N * N * fwd(2nd order)
    // template <typename F, typename S>
    // void
    // hessian_times_vector(const F& f,
    //                      const Eigen::Matrix<S,Eigen::Dynamic,1>& x,
    //                      S& fx,
    //                      const Eigen::Matrix<S,Eigen::Dynamic,1>& v,
    //                      Eigen::Matrix<S,Eigen::Dynamic,1>& Hv);

    // // 1 * (fwd + rev)
    // template <typename F>
    // void
    // hessian_times_vector_fast(const F& f,
    //                           const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
    //                           double& fx,
    //                           const Eigen::Matrix<double,Eigen::Dynamic,1>& v,
    //                           double& v_dot_grad_fx,
    //                           Eigen::Matrix<double,Eigen::Dynamic,1>& Hv);
    // // N * (fwd(2) + bk)
    // template <typename F>
    // void
    // gradient_trace_matrix_times_hessian(const F& f,
    //                                     const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
    //                                     double& fx,
    //                                     const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& M,
    //                                     Eigen::matrix<double,Eigen::Dynamic,1>& grad_tr_MH);
                                
  }
}

#endif
