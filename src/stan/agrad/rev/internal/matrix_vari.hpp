#ifndef STAN__AGRAD__REV__INTERNAL__MATRIX_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__MATRIX_VARI_HPP

#include <stan/agrad/rev/matrix/Eigen_NumTraits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {

    class op_matrix_vari : public vari {
    protected:
      const size_t size_;
      vari** vis_;
    public:
      template <int R, int C>
      op_matrix_vari(double f, const Eigen::Matrix<stan::agrad::var,R,C>& vs) :
        vari(f),
        size_(vs.size()) {

        vis_ = (vari**) operator new(sizeof(vari*) * vs.size()); 
        for (int i = 0; i < vs.size(); ++i)
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
