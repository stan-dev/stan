#ifndef STAN__AGRAD__AUTO_DIFF_HPP
#define STAN__AGRAD__AUTO_DIFF_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>

namespace stan {
  
  namespace agrad {

    /**
     * Return the derivative of the specified univariate function at
     * the specified argument.
     *
     * @tparam T Argument type
     * @tparam F Function type
     * @param[in] f Function
     * @param[in] x Argument
     * @param[out] fx Value of function applied to argument
     * @param[out] dfx_dx Value of derivative
     */
    template <typename T, typename F>
    void
    derivative(const F& f,
               const T& x,
               T& fx,
               T& dfx_dx)  {
      fvar<T> x_fvar = fvar<T>(x,1.0);
      fvar<T> fx_fvar = f(x_fvar); 
      fx = fx_fvar.val_;
      dfx_dx = fx_fvar.d_;
    }

    /**
     * Return the partial derivative of the specified multiivariate
     * function at the specified argument.
     *
     * @tparam T Argument type
     * @tparam F Function type
     * @param f Function
     * @param[in] x Argument vector
     * @param[in] n Index of argument with which to take derivative
     * @param[out] fx Value of function applied to argument
     * @param[out] dfx_dxn Value of partial derivative
     */
    template <typename T, typename F>
    void
    partial_derivative(const F& f,
                       const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                       int n,
                       T& fx,
                       T& dfx_dxn) {
      Eigen::Matrix<fvar<T>,Eigen::Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_fvar(i) = fvar<T>(x(i),i==n);
      fvar<T> fx_fvar = f(x_fvar);
      fx = fx_fvar.val_;
      dfx_dxn = fx_fvar.d_;
    }

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
      start_nested();
      try {
        Eigen::Matrix<var,Eigen::Dynamic,1> x_var(x.size());
        for (int i = 0; i < x.size(); ++i)
          x_var(i) = x(i);
        var fx_var = f(x_var);
        fx = fx_var.val();
        grad_fx.resize(x.size());
        stan::agrad::grad(fx_var.vi_);
        for (int i = 0; i < x.size(); ++i)
          grad_fx(i) = x_var(i).adj();
      } catch (const std::exception& /*e*/) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }
    template <typename T, typename F>
    void
    gradient(const F& f,
             const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
             T& fx,
             Eigen::Matrix<T,Eigen::Dynamic,1>& grad_fx) {
      Eigen::Matrix<fvar<T>,Eigen::Dynamic,1> x_fvar(x.size());
      grad_fx.resize(x.size());
      for (int i = 0; i < x.size(); ++i) {
        for (int k = 0; k < x.size(); ++k)
          x_fvar(k) = fvar<T>(x(k),k==i);
        fvar<T> fx_fvar = f(x_fvar);
        if (i == 0) fx = fx_fvar.val_;
        grad_fx(i) = fx_fvar.d_;
      }
    }


    template <typename F>
    void
    jacobian(const F& f,
             const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
             Eigen::Matrix<double,Eigen::Dynamic,1>& fx,
             Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& J) {
      using Eigen::Matrix;  using Eigen::Dynamic;
      using stan::agrad::var;
      start_nested();
      try {
        Matrix<var,Dynamic,1> x_var(x.size());
        for (int k = 0; k < x.size(); ++k)
          x_var(k) = x(k);
        Matrix<var,Dynamic,1> fx_var = f(x_var);
        fx.resize(fx_var.size());
        for (int i = 0; i < fx_var.size(); ++i)
          fx(i) = fx_var(i).val(); 
        J.resize(x.size(), fx_var.size());
        for (int i = 0; i < fx_var.size(); ++i) {
          if (i > 0)
            set_zero_all_adjoints();
          grad(fx_var(i).vi_);
          for (int k = 0; k < x.size(); ++k)
            J(k,i) = x_var(k).adj();
        }
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }
    template <typename T, typename F>
    void
    jacobian(const F& f,
             const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
             Eigen::Matrix<T,Eigen::Dynamic,1>& fx,
             Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& J) {
      using Eigen::Matrix;  using Eigen::Dynamic;
      using stan::agrad::fvar;
      Matrix<fvar<T>,Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i) {
        for (int k = 0; k < x.size(); ++k)
          x_fvar(k) = fvar<T>(x(k), i == k);
        Matrix<fvar<T>,Dynamic,1> fx_fvar 
          = f(x_fvar);
        if (i == 0) {
          J.resize(x.size(),fx_fvar.size());
          fx.resize(fx_fvar.size());
          for (int k = 0; k < fx_fvar.size(); ++k)
            fx(k) = fx_fvar(k).val_;
        }
        for (int k = 0; k < fx_fvar.size(); ++k) {
          J(i,k) = fx_fvar(k).d_;
        }
      }
    }
      

    // time O(N^2);  space O(N^2)
    template <typename F>
    void
    hessian(const F& f,
            const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
            double& fx,
            Eigen::Matrix<double,Eigen::Dynamic,1>& grad,
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& H) {
      start_nested();
      try {
        H.resize(x.size(), x.size());
        grad.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
          Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> x_fvar(x.size());
          for (int j = 0; j < x.size(); ++j) 
            x_fvar(j) = fvar<var>(x(j),i==j);
          fvar<var> fx_fvar = f(x_fvar);
          grad(i) = fx_fvar.d_.val();
          if (i == 0) fx = fx_fvar.val_.val();
          stan::agrad::grad(fx_fvar.d_.vi_);
          for (int j = 0; j < x.size(); ++j)
            H(i,j) = x_fvar(j).val_.adj();
        }
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }
    // time O(N^3);  space O(N^2)
    template <typename T, typename F>
    void
    hessian(const F& f,
            const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
            T& fx,
            Eigen::Matrix<T,Eigen::Dynamic,1>& grad,
            Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& H) {
      H.resize(x.size(), x.size());
      grad.resize(x.size());
      Eigen::Matrix<fvar<fvar<T> >,Eigen::Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i) {
        for (int j = i; j < x.size(); ++j) {
          for (int k = 0; k < x.size(); ++k)
            x_fvar(k) = fvar<fvar<T> >(fvar<T>(x(k),j==k), 
                                       fvar<T>(i==k,0));
          fvar<fvar<T> > fx_fvar = f(x_fvar);
          if (j == 0) 
            fx = fx_fvar.val_.val_;
          if (i == j)
            grad(i) = fx_fvar.d_.val_;
          H(i,j) = fx_fvar.d_.d_;
          H(j,i) = H(i,j);
        }
      }
    }


    // aka directional derivative (not length normalized)
    // T2 must be assignable to T1
    template <typename T1, typename T2, typename F>
    void
    gradient_dot_vector(const F& f,
                        const Eigen::Matrix<T1,Eigen::Dynamic,1>& x,
                        const Eigen::Matrix<T2,Eigen::Dynamic,1>& v,
                        T1& fx,
                        T1& grad_fx_dot_v) {
      using stan::agrad::fvar;
      using stan::agrad::var;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      Matrix<fvar<T1>,Dynamic,1> x_fvar(x.size());
      for (int i = 0; i < x.size(); ++i)
        x_fvar(i) = fvar<T1>(x(i),v(i));
      fvar<T1> fx_fvar = f(x_fvar);
      fx = fx_fvar.val_;
      grad_fx_dot_v = fx_fvar.d_;
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
      using Eigen::Matrix; 
      using Eigen::Dynamic;
      start_nested();
      try {
        Matrix<var,Dynamic,1> x_var(x.size());
        for (int i = 0; i < x_var.size(); ++i)
          x_var(i) = x(i);
        var fx_var;
        var grad_fx_var_dot_v;
        gradient_dot_vector(f,x_var,v,fx_var,grad_fx_var_dot_v);
        fx = fx_var.val();
        stan::agrad::grad(grad_fx_var_dot_v.vi_);
        Hv.resize(x.size());
        for (int i = 0; i < x.size(); ++i) 
          Hv(i) = x_var(i).adj();
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }
    template <typename T, typename F>
    void
    hessian_times_vector(const F& f,
                         const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                         const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
                         T& fx,
                         Eigen::Matrix<T,Eigen::Dynamic,1>& Hv) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      Matrix<T,Dynamic,1> grad;
      Matrix<T,Dynamic,Dynamic> H;
      hessian(f,x,fx,grad,H);
      Hv = H * v;
    }

    // FIXME: add other results that are easy to extract
    // // N * (fwd(2) + bk)
    template <typename F>
    void
    grad_tr_mat_times_hessian(
                 const F& f,
                 const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                 const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& M,
                 Eigen::Matrix<double,Eigen::Dynamic,1>& grad_tr_MH) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      start_nested();
      try {
      
        grad_tr_MH.resize(x.size());

        Matrix<var,Dynamic,1> x_var(x.size());
        for (int i = 0; i < x.size(); ++i)
          x_var(i) = x(i);

        Matrix<fvar<var>,Dynamic,1> x_fvar(x.size());
      
        var sum(0.0);
        Matrix<double,Dynamic,1> M_n(x.size());
        for (int n = 0; n < x.size(); ++n) {
          for (int k = 0; k < x.size(); ++k)
            M_n(k) = M(n,k);
          for (int k = 0; k < x.size(); ++k)
            x_fvar(k) = fvar<var>(x_var(k), k == n);
          fvar<var> fx;
          fvar<var> grad_fx_dot_v;
          gradient_dot_vector<fvar<var>,double>(f,x_fvar,M_n,fx,grad_fx_dot_v);
          sum += grad_fx_dot_v.d_;
        }

        stan::agrad::grad(sum.vi_);
        for (int i = 0; i < x.size(); ++i)
          grad_tr_MH(i) = x_var(i).adj();
      } catch (const std::exception& e) {
        stan::agrad::recover_memory_nested();
        throw;
      }
      stan::agrad::recover_memory_nested();
    }

  }
}
#endif
