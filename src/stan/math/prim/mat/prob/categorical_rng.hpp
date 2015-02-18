#ifndef STAN__MATH__PRIM__MAT__PROB__CATEGORICAL_RNG_HPP
#define STAN__MATH__PRIM__MAT__PROB__CATEGORICAL_RNG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/mat/err/check_simplex.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline int
    categorical_rng(const Eigen::Matrix<double,Eigen::Dynamic,1>& theta,
                    RNG& rng) {
      using boost::variate_generator;
      using boost::uniform_01;
      using stan::math::check_simplex;

      static const char* function("stan::prob::categorical_rng");

      check_simplex(function, "Probabilities parameter", theta);

      variate_generator<RNG&, uniform_01<> >
        uniform01_rng(rng, uniform_01<>());
      
      Eigen::VectorXd index(theta.rows());
      index.setZero();

      for(int i = 0; i < theta.rows(); i++) {
        for(int j = i; j < theta.rows(); j++)
          index(j) += theta(i,0);
      }

      double c = uniform01_rng();
      int b = 0;
      while (c > index(b,0))
        b++;
      return b + 1;
    }
  }
}
#endif
