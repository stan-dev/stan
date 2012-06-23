#ifndef __STAN__AGRAD__PARTIALS_VARI_HPP__
#define __STAN__AGRAD__PARTIALS_VARI_HPP__

#include <vector>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>

namespace stan {

  namespace agrad {



    /**
     * A variable implementation that stores a single operand and its
     * derivative with respect to the variable.
     */
    class partials1_vari : public vari {
    private:
      vari* operand1_;
    public: 
      const double partial1_;
      /**
       * Construct a variable implementation with the specified value,
       * operand and derivative.
       * @param value Value of variable.
       * @param operand1 Implementation of first operand.
       * @param partial1 Derivative of variable with respect to first operand.
       */
      partials1_vari(double value, vari* operand1, double partial1) 
        : vari(value), operand1_(operand1), partial1_(partial1) 
      { }
      /**
       * Apply chain rule by incrementing this variable's operand's
       * adjoint by the product of this variable's adjoint and
       * the stored partial.
       */
      void chain() {
        operand1_->adj_ += adj_ * partial1_;
      }
    };

    /**
     * A variable implementation that stores two operands and their
     * derivatives with respect to the variable.
     */
    class partials2_vari : public vari {
    private:
      vari* operand1_;      const double partial1_;
      vari* operand2_;      const double partial2_;
    public:
      /**
       * Construct a variable implementation with the specified value,
       * operands and derivatives.
       * @param value Value of variable.
       * @param operand1 Implementation of first operand.
       * @param partial1 Derivative of variable with respect to first operand.
       * @param operand2 Implementation of second operand.
       * @param partial2 Derivative of variable with respect to second operand.
       */
      partials2_vari(double value, 
                     vari* operand1, double partial1,
                     vari* operand2, double partial2) 
        : vari(value), 
          operand1_(operand1), partial1_(partial1),
          operand2_(operand2), partial2_(partial2)
      { }
      /**
       * Apply chain rule by incrementing each variable's operand's
       * adjoint by the product of this variable's adjoint and
       * the derivative with respect to the operand.
       */
      void chain() {
        operand1_->adj_ += adj_ * partial1_;
        operand2_->adj_ += adj_ * partial2_;
      }
    };

    /**
     * A variable implementation that stores three operands and their
     * derivatives with respect to the variable.
     */
    class partials3_vari : public vari {
    private:
      vari* operand1_;  const double partial1_;
      vari* operand2_;  const double partial2_;
      vari* operand3_;  const double partial3_;
    public:
      /**
       * Construct a variable implementation with the specified value,
       * operands and derivatives.
       * @param value Value of variable.
       * @param operand1 Implementation of first operand.
       * @param partial1 Derivative of variable with respect to first operand.
       * @param operand2 Implementation of second operand.
       * @param partial2 Derivative of variable with respect to second operand.
       * @param operand3 Implementation of second operand.
       * @param partial3 Derivative of variable with respect to third operand.
       */
      partials3_vari(double value, 
                     vari* operand1, double partial1,
                     vari* operand2, double partial2,
                     vari* operand3, double partial3)
        : vari(value), 
          operand1_(operand1), partial1_(partial1),
          operand2_(operand2), partial2_(partial2),
          operand3_(operand3), partial3_(partial3)
      { }
      /**
       * Apply chain rule by incrementing each variable's operand's
       * adjoint by the product of this variable's adjoint and
       * the derivative with respect to the operand.
       */
      void chain() {
        operand1_->adj_ += adj_ * partial1_;
        operand2_->adj_ += adj_ * partial2_;
        operand3_->adj_ += adj_ * partial3_;
      }
    };

    vari** to_array(const std::vector<var>& xvec) {
      vari** xs = static_cast<vari**>(memalloc_.alloc(xvec.size() * sizeof(vari*)));
      for (size_t n = 0; n < xvec.size(); ++n)
        xs[n] = xvec[n].vi_;
      return xs;
    }

    double* to_array(const std::vector<double>& xvec) {
      double* xs = static_cast<double*>(memalloc_.alloc(xvec.size() * sizeof(double)));
      for (size_t n = 0; n < xvec.size(); ++n)
        xs[n] = xvec[n];
      return xs;
    }


    class partials1s_2_vari : public vari {
    private:
      const size_t N_;
      vari** operands1_;   const double* partials1_;
      vari* operand2_;     const double partial2_;
      vari* operand3_;     const double partial3_;
    public: 
      partials1s_2_vari(
                double value,
                const std::vector<var>& operands1, const std::vector<double>& partials1,
                vari* operand2, double partial2,
                vari* operand3, double partial3)
        : vari(value),
          N_(operands1.size()),
          operands1_(to_array(operands1)), partials1_(to_array(partials1)),
          operand2_(operand2), partial2_(partial2),
          operand3_(operand3), partial3_(partial3) {
      }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands1_[n]->adj_ += adj_ * partials1_[n];
        operand2_->adj_ += adj_ * partial2_;
        operand3_->adj_ += adj_ * partial3_;
      }
    };

    class partials1s_1_vari : public vari {
    private:
      const size_t N_;
      vari** operands1_;   const double* partials1_;
      vari* operand2_;     const double partial2_;
    public: 
      partials1s_1_vari(
                double value,
                const std::vector<var>& operands1, const std::vector<double>& partials1,
                vari* operand2, double partial2)
        : vari(value),
          N_(operands1.size()),
          operands1_(to_array(operands1)), partials1_(to_array(partials1)),
          operand2_(operand2), partial2_(partial2) {
      }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands1_[n]->adj_ += adj_ * partials1_[n];
        operand2_->adj_ += adj_ * partial2_;
      }
    };

    class partials1s_vari : public vari {
    private:
      const size_t N_;
      vari** operands1_;   const double* partials1_;
    public: 
      partials1s_vari(
              double value,
              const std::vector<var>& operands1, const std::vector<double>& partials1)
        : vari(value),
          N_(operands1.size()),
          operands1_(to_array(operands1)), partials1_(to_array(partials1)) {
      }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands1_[n]->adj_ += adj_ * partials1_[n];
      }
    };

    // class partials2s_1_vari : public vari {
    // private:
    //   const size_t N_;
    //   vari** operands1_;   const double* partials1_;
    //   vari** operands2_;   const double* partials2_;
    //   vari* operand3_;     const double partial3_;
    // public: 
    //   partials2s_1_vari(double value,
    //                     size_t N,
    //                     vari** operands1, double* partials1,
    //                     vari** operands2, double* partials2,
    //                     vari* operand3, double partial3)
    //     : vari(value),
    //       N_(N),
    //       operands1_(operands1), partials1_(partials1),
    //       operands2_(operands2), partials2_(partials2),
    //       operand3_(operand3), partial3_(partial3) {
    //   }
    //   void chain() {
    //     for (size_t n = 0; n < N_; ++n)
    //       operands1_[n]->adj_ += adj_ * partials1_[n];
    //     for (size_t n = 0; n < N_; ++n)
    //       operands2_[n]->adj_ += adj_ * partials2_[n];
    //     operand3_->adj_ += adj_ * partial3_;
    //   }
    // };

    // class partials2s_vari : public vari {
    // private:
    //   const size_t N_;
    //   vari** operands1_;   const double* partials1_;
    //   vari** operands2_;   const double* partials2_;
    // public: 
    //   partials2s_vari(double value,
    //                   size_t N,
    //                   vari** operands1, double* partials1,
    //                   vari** operands2, double* partials2)
    //     : vari(value),
    //       N_(N),
    //       operands1_(operands1), partials1_(partials1),
    //       operands2_(operands2), partials2_(partials2) {
    //   }
    //   void chain() {
    //     for (size_t n = 0; n < N_; ++n)
    //       operands1_[n]->adj_ += adj_ * partials1_[n];
    //     for (size_t n = 0; n < N_; ++n)
    //       operands2_[n]->adj_ += adj_ * partials2_[n];
    //   }
    // };

    // class partials3s_vari : public vari {
    // private:
    //   const size_t N_;
    //   vari** operands1_;   const double* partials1_;
    //   vari** operands2_;   const double* partials2_;
    //   vari** operands3_;   const double* partials3_;
    // public: 
    //   partials3s_vari(double value,
    //                   size_t N,
    //                   vari** operands1, double* partials1,
    //                   vari** operands2, double* partials2,
    //                   vari** operands3, double* partials3)
    //     : vari(value),
    //       N_(N),
    //       operands1_(operands1), partials1_(partials1),
    //       operands2_(operands2), partials2_(partials2),
    //       operands3_(operands3), partials3_(partials3) {
    //   }
    //   void chain() {
    //     for (size_t n = 0; n < N_; ++n)
    //       operands1_[n]->adj_ += adj_ * partials1_[n];
    //     for (size_t n = 0; n < N_; ++n)
    //       operands2_[n]->adj_ += adj_ * partials2_[n];
    //     for (size_t n = 0; n < N_; ++n)
    //       operands3_[n]->adj_ += adj_ * partials3_[n];
    //   }
    // };


    // 111
    inline agrad::var simple_var(
                         double v,
                         const std::vector<agrad::var>& y1, const std::vector<double> dy1, 
                         const agrad::var& y2, double dy2,
                         const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials1s_2_vari(v,
                                                     y1, dy1,
                                                     y2.vi_, dy2,
                                                     y3.vi_, dy3));
    }

    // 110
    template <typename T3>
    inline agrad::var simple_var(
                         double v,
                         const std::vector<agrad::var>& y1, const std::vector<double> dy1, 
                         const agrad::var& y2, double dy2,
                         const T3& /*y3*/, const T3& /*dy3*/) {
      return agrad::var(new agrad::partials1s_1_vari(v,
                                                     y1, dy1,
                                                     y2.vi_, dy2));
    }

    // 101
    template <typename T2>
    inline agrad::var simple_var(
                         double v,
                         const std::vector<agrad::var>& y1, const std::vector<double> dy1, 
                         const T2& /*y2*/, const T2& /*dy2*/,
                         const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials1s_1_vari(v,
                                                     y1, dy1,
                                                     y3.vi_, dy3));
    }

    // 100
    template <typename T2, typename T3>
    inline agrad::var simple_var(
                         double v,
                         const std::vector<agrad::var>& y1, const std::vector<double> dy1, 
                         const T2& /*y2*/, const T2& /*dy2*/,
                         const T3& /*y3*/, const T3& /*dy3*/) {
      return agrad::var(new agrad::partials1s_vari(v,
                                                   y1, dy1));
    }





    /**
     * Return an instance of <code>partials3_vari</code> with the
     * specified values, operands and partial derivatives.
     *
     * @param v Value of variable.
     * @param y1 First operand.
     * @param dy1 Derivative of variable with respect to first operand.
     * @param y2 Second operand.
     * @param dy2 Derivative of variable with respect to second operand.
     * @param y3 Third operand.
     * @param dy3 Derivative of variable with respect to third operand.
     */
    inline agrad::var simple_var(double v, 
                                 const agrad::var& y1, double dy1, 
                                 const agrad::var& y2, double dy2,
                                 const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials3_vari(v,
                                                  y1.vi_, dy1,
                                                  y2.vi_, dy2,
                                                  y3.vi_, dy3));
    }

    /**
     * Return an instance of <code>partials2_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y1 First operand.
     * @param dy1 Derivative of variable with respect to first operand.
     * @param y2 Second operand.
     * @param dy2 Derivative of variable with respect to second operand.
     */
    template <typename T3>
    inline agrad::var simple_var(double v, 
                                 const agrad::var& y1, double dy1, 
                                 const agrad::var& y2, double dy2,
                                 T3 /*y3*/, T3 /*dy3*/) {
      return agrad::var(new agrad::partials2_vari(v,
                                                  y1.vi_, dy1,
                                                  y2.vi_, dy2));
    }

    /**
     * Return an instance of <code>partials2_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y1 First operand.
     * @param dy1 Derivative of variable with respect to first operand.
     * @param y3 Third operand.
     * @param dy3 Derivative of variable with respect to third operand.
     */
    template <typename T2>
    inline agrad::var simple_var(double v, 
                                 const agrad::var& y1, double dy1, 
                                 T2 /*y2*/, T2 /*dy2*/,
                                 const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials2_vari(v,
                                                  y1.vi_, dy1,
                                                  y3.vi_, dy3));
    }

    /**
     * Return an instance of <code>partials1_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y1 First operand.
     * @param dy1 Derivative of variable with respect to first operand.
     */
    template <typename T2, typename T3>
    inline agrad::var simple_var(double v, 
                                 const agrad::var& y1, double dy1, 
                                 T2 /*y2*/, T2 /*dy2*/,
                                 T3 /*y3*/, T3 /*dy3*/) {
      return agrad::var(new agrad::partials1_vari(v,
                                                  y1.vi_, dy1));
    }

    /**
     * Return an instance of <code>partials2_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y2 Second operand.
     * @param dy2 Derivative of variable with respect to second operand.
     * @param y3 Third operand.
     * @param dy3 Derivative of variable with respect to third operand.
     */
    template <typename T1>
    inline agrad::var simple_var(double v, 
                                 T1 /*y1*/, T1 /*dy*/, 
                                 const agrad::var& y2, double dy2,
                                 const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials2_vari(v,
                                                  y2.vi_, dy2,
                                                  y3.vi_, dy3));
    }

    /**
     * Return an instance of <code>partials1_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y2 Second operand.
     * @param dy2 Derivative of variable with respect to second operand.
     */
    template <typename T1, typename T3>
    inline agrad::var simple_var(double v, 
                                 T1 /*y1*/, T1 /*dy1*/, 
                                 const agrad::var& y2, double dy2,
                                 T3 /*y3*/, T3 /*dy3*/) {
      return agrad::var(new agrad::partials1_vari(v,
                                                  y2.vi_, dy2));
    }

    /**
     * Return an instance of <code>partials1_vari</code> with the
     * specified values, operands and partial derivatives, ignoring
     * the undocumented parameter positions.
     *
     * @param v Value of variable.
     * @param y3 Third operand.
     * @param dy3 Derivative of variable with respect to third operand.
     */
    template <typename T1, typename T2>
    inline agrad::var simple_var(double v, 
                                 T1 /*y1*/, T1 /*dy1*/, 
                                 T2 /*y2*/, T2 /*dy2*/,
                                 const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials1_vari(v,
                                                  y3.vi_, dy3));
    }


  } 
} 


#endif
