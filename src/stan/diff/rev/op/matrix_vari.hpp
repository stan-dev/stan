#ifndef __STAN__DIFF__REV__OP__MATRIX_VARI_HPP__
#define __STAN__DIFF__REV__OP__MATRIX_VARI_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {

    class op_matrix_vari : public vari {
    protected:
      const size_t size_;
      vari** vis_;
    public:
      template <int R, int C>
      op_matrix_vari(double f, const Eigen::Matrix<stan::diff::var,R,C>& vs) :
        vari(f),
        size_(vs.size()) {

        vis_ = (vari**) operator new(sizeof(vari*) * vs.size()); 
        for (size_t i = 0; i < vs.size(); ++i)
          vis_[i] = vs(i).vi_;
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
