#ifndef __STAN__AGRAD__PARTIALS_VARI_HPP__
#define __STAN__AGRAD__PARTIALS_VARI_HPP__

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
    inline agrad::var simple_var(double v, 
                          const agrad::var& y1, double dy1, 
                          const agrad::var& y2, double dy2,
                          double /*y3*/, double /*dy3*/) {
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
    inline agrad::var simple_var(double v, 
                          const agrad::var& y1, double dy1, 
                          double /*y2*/, double /*dy2*/,
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
    inline agrad::var simple_var(double v, 
                          const agrad::var& y1, double dy1, 
                          double /*y2*/, double /*dy2*/,
                          double y3, double dy3) {
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
    inline agrad::var simple_var(double v, 
                          double /*y1*/, double /*dy*/, 
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
    inline agrad::var simple_var(double v, 
                          double /*y1*/, double /*dy1*/, 
                          const agrad::var& y2, double dy2,
                          double /*y3*/, double /*dy3*/) {
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
    inline agrad::var simple_var(double v, 
                          double /*y1*/, double /*dy1*/, 
                          double /*y2*/, double /*dy2*/,
                          const agrad::var& y3, double dy3) {
      return agrad::var(new agrad::partials1_vari(v,
                                    y3.vi_, dy3));
    }


  } 
} 


#endif
