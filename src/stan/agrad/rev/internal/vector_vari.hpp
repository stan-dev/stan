#ifndef STAN__AGRAD__REV__INTERNAL__VECTOR_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__VECTOR_VARI_HPP

#include <vector>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>

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
        vis_ = (vari**) operator new(sizeof(vari*) * vs.size()); 
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
