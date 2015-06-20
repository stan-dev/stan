#ifndef STAN_VARIATIONAL_BASE_FAMILY_HPP
#define STAN_VARIATIONAL_BASE_FAMILY_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>

#include <stan/model/util.hpp>

#include <vector>
#include <ostream>

namespace stan {

  namespace variational {

    class base_family {
    public:
      // Constructors
      base_family() {};

      // Operations
      base_family square() const;
      base_family sqrt() const;

      // Compound assignment operators
      base_family operator=(const base_family& rhs);
      base_family operator+=(const base_family& rhs);
      base_family operator/=(const base_family& rhs);
      base_family operator+=(double scalar);
      base_family operator*=(double scalar);

      // Distribution-based operations
      const Eigen::VectorXd& mean() const;
      double entropy() const;
      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const;
      template <class BaseRNG>
      Eigen::VectorXd sample(BaseRNG& rng) const;
      template <class M, class BaseRNG>
      void calc_grad(base_family& params_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream);
    };

    // Arithmetic operators
    base_family operator+(base_family lhs, const base_family& rhs);
    base_family operator+(double scalar, base_family rhs);
    base_family operator*(double scalar, base_family rhs);
    base_family operator/(base_family lhs, const base_family& rhs);
  }  // variational
}  // stan

#endif
