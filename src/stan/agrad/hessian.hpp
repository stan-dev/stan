#ifndef STAN__AGRAD__HESSIAN_HPP
#define STAN__AGRAD__HESSIAN_HPP

#include <vector>

#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace agrad {

    // f: vector<T> -> T
    // Hv = H(f(x)) * v;  return f(x)

    template <class F>
    double
    hessian(const F& f,
            const std::vector<double>& x,
            const std::vector<double>& v,
            std::vector<double>& Hv) {

      using stan::agrad::var;
      using stan::agrad::fvar;
      
      std::vector<var> x_var(x.size());
      for (int i = 0; i < x_var.size(); ++i)
        x_var[i] = x[i];

      std::vector<fvar<var> > x_fvar(x.size());
      for (int i = 0; i < x_fvar.size(); ++i)
        x_fvar[i] = fvar<var>(x_var[i], v[i]);  
      fvar<var> y = f(x_fvar);
      y.d_.grad(x_var,Hv);
      return y.val_.val();
    }

  }
}
            

#endif
