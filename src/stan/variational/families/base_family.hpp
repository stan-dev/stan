#ifndef STAN_VARIATIONAL_BASE_FAMILY_HPP
#define STAN_VARIATIONAL_BASE_FAMILY_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>
#include <ostream>

namespace stan {

  namespace variational {

    class base_family {
    public:
      base_family() {}

      base_family square() const;
      base_family sqrt() const;

      base_family operator=(const base_family& rhs);
      base_family operator+=(const base_family& rhs);
      base_family operator/=(const base_family& rhs);
      base_family operator+=(double scalar);
      base_family operator*=(double scalar);

      const Eigen::VectorXd& mean() const;
      double entropy() const;
      template <class BaseRNG>
      void sample(BaseRNG& rng, Eigen::VectorXd& eta) const;
      template <class M, class BaseRNG>
      void calc_grad(base_family& elbo_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream) const;

    protected:
      void write_error_msg_(std::ostream* error_msgs,
                            const std::exception& e) const {
        if (!error_msgs) {
          return;
        }

        *error_msgs
          << std::endl
          << "Informational Message: The current gradient evaluation "
          << "of the ELBO is ignored because of the following issue:"
          << std::endl
          << e.what() << std::endl
          << "If this warning occurs often then your model may be "
          << "either severely ill-conditioned or misspecified."
          << std::endl;
      }
    };

    base_family operator+(base_family lhs, const base_family& rhs);
    base_family operator/(base_family lhs, const base_family& rhs);
    base_family operator+(double scalar, base_family rhs);
    base_family operator*(double scalar, base_family rhs);

  }  // variational

}  // stan

#endif
