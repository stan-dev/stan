#ifndef TEST_UNIT_MODEL_TEST_MODEL_HPP
#define TEST_UNIT_MODEL_TEST_MODEL_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/io/reader.hpp>

class TestModel_uniform_01 {
public:
  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r,
             std::vector<int>& params_i,
             std::ostream* pstream_ = 0) const {
    T lp(0.0);
    stan::math::accumulator<T> lp_accum;
    
    // model parameters
    stan::io::reader<T> in(params_r, params_i);
    
    T y;
    if (jacobian)
      y = in.scalar_lub_constrain(0,1,lp);
    else
      y = in.scalar_lub_constrain(0,1);
    
    lp_accum.add(stan::math::uniform_log<propto>(y, 0, 1));
    lp_accum.add(lp);

    return lp_accum.sum();
  }
};

#endif
