#ifndef __STAN__DIFF__REV__OP__VECTOR_VARI_HPP__
#define __STAN__DIFF__REV__OP__VECTOR_VARI_HPP__

#include <vector>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {

    class op_vector_vari : public vari {
    protected:
      const size_t size_;
      vari** vis_;
    public:
      op_vector_vari(double f, const std::vector<stan::diff::var>& vs) :
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
