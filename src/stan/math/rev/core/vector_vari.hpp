#ifndef STAN__MATH__REV__CORE__VECTOR_VARI_HPP
#define STAN__MATH__REV__CORE__VECTOR_VARI_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/vari.hpp>
#include <vector>

namespace stan {
  namespace agrad {

    class op_vector_vari : public vari {
    protected:
      const size_t size_;
      vari** vis_;
    public:
      op_vector_vari(double f, const std::vector<stan::agrad::var>& vs) :
        vari(f),
        size_(vs.size()) {
        vis_ = reinterpret_cast<vari**>
          (operator new(sizeof(vari*) * vs.size()));
        for (size_t i = 0; i < vs.size(); ++i)
          vis_[i] = vs[i].vi_;
      }
      vari* operator[](size_t n) const {
        return vis_[n];
      }
      size_t size() {
        return size_;
      }
    };

  }
}
#endif
