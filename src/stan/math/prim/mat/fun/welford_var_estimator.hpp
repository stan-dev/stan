#ifndef STAN__MATH__PRIM__MAT__FUN__WELFORD_VAR_ESTIMATOR_HPP
#define STAN__MATH__PRIM__MAT__FUN__WELFORD_VAR_ESTIMATOR_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  namespace prob {

    class welford_var_estimator {

    public:

      welford_var_estimator(int n): _m(Eigen::VectorXd::Zero(n)),
                                    _m2(Eigen::VectorXd::Zero(n))
      { restart(); }

      void restart() {
        _num_samples = 0;
        _m.setZero();
        _m2.setZero();
      }

      void add_sample(const Eigen::VectorXd& q) {

        ++_num_samples;

        Eigen::VectorXd delta(q - _m);
        _m  += delta / _num_samples;
        _m2 += delta.cwiseProduct(q - _m);

      }

      int num_samples() { return _num_samples; }

      void sample_mean(Eigen::VectorXd& mean) { mean = _m; }

      void sample_variance(Eigen::VectorXd& var) {
        if(_num_samples > 1)
          var = _m2 / (_num_samples - 1.0);
      }

    protected:

      double _num_samples;

      Eigen::VectorXd _m;
      Eigen::VectorXd _m2;

    };

  } // prob

} // stan


#endif
