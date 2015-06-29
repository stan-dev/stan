#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_LOWER_UPPER_BOUND_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_LOWER_UPPER_BOUND_HPP

#include <vector>

#include <stan/math/prim/scal/fun/lub_constrain.hpp>
#include <stan/math/prim/scal/fun/lub_free.hpp>

#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_lower_upper_bound: public scalar_transform<T> {
    private:
      T lower_bound_;
      T upper_bound_;
      
    public:
      scalar_lower_upper_bound(T lower_bound, T upper_bound):
        lower_bound_(lower_bound), upper_bound_(upper_bound) {}
      
      void unconstrain(T input, vector<T>& output) {
        output.push_back(stan::prob::lub_free(input,
                                              lower_bound_,
                                              upper_bound_));
      }
      
      T unconstrain(T input) {
        return stan::prob::lub_free(input, lower_bound_, upper_bound_);
      }
      
      void constrain(T input, T& output) {
        output = stan::prob::lub_constraint(input,
                                            lower_bound_,
                                            upper_bound_);
      }
      
      void constrain(T input, T& output, T& lp) {
        output = stan::prob::lub_constraint(input,
                                            lower_bound_,
                                            upper_bound_,
                                            lp);
      }
      
      T constrain(T input) {
        return stan::prob::lub_constraint(input,
                                          lower_bound_,
                                          upper_bound_);
      }
      
      T constrain(T input, T& lp) {
        return stan::prob::lub_constraint(input,
                                          lower_bound_,
                                          upper_bound_,
                                          lp);
      }
    }

  }  // io
}  // stan

#endif
