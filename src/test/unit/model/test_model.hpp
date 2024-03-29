#ifndef TEST_UNIT_MODEL_TEST_MODEL_HPP
#define TEST_UNIT_MODEL_TEST_MODEL_HPP

#include <stan/math/prim.hpp>
#include <stan/io/deserializer.hpp>

class TestModel_uniform_01 {
 public:
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__, std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;

    // model parameters
    stan::io::deserializer<T__> in__(params_r__, params_i__);

    T__ y;
    y = in__.template read_constrain_lub<T__, jacobian__>(0, 1, lp__);

    lp_accum__.add(stan::math::uniform_lpdf<propto__>(y, 0, 1));
    lp_accum__.add(lp__);

    return lp_accum__.sum();
  }
};

#endif
