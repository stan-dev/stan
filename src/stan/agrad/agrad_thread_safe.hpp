#ifndef STAN__AGRAD__AGRAD_HPP
#define STAN__AGRAD__AGRAD_HPP

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include "stan/memory/stack_alloc.hpp"

// FIXME:  Should include common defs, not this huge cut-and-paste!

namespace stan {

  namespace agrad {

    class vari;

    namespace {
      struct var_allocator {
        std::vector<vari*> var_stack_; 
        memory::stack_alloc memalloc_;
        inline void* alloc(size_t nbytes) {
          return memalloc_.alloc(nbytes);
        }
        inline void recover() {
          var_stack_.resize(0);
          memalloc_.recover_all();
        }
        inline void free() {
          memalloc_.free_all();
        }
      };
#ifdef AGRAD_THREAD_SAFE      
      __thread
#endif
      var_allocator* allocator_;
    }

    /**
     * The variable implementation base class.
     *
     * A variable implementation is constructed with a constant
     * value.  It also stores the adjoint for storing the partial
     * derivative with respect to the root of the derivative tree.
   
     * The chain() method applies the chain rule.  Concrete extensions
     * of this class will represent base variables or the result
     * of operations such as addition or subtraction.  These extended
     * classes will store operand variables and propagate derivative
     * information via an implementation of chain().
     */
    class vari {
    private:
      friend class var;

    public: 

      /**
       * The value of this variable.
       */
      const double val_;

      /**
       * The adjoint of this variable, which is the partial derivative
       * of this variable with respect to the root variable.
       */
      double adj_;
      
      /**
       * Construct a variable implementation from a value.  The
       * adjoint is initialized to zero. 
       *
       * All constructed variables are added to the stack.  Variables
       * should be constructed before variables on which they depend
       * to insure proper partial derivative propagation.  During
       * derivative propagation, the chain() method of each variable
       * will be called in the reverse order of construction.
       *
       * @param x Value of the constructed variable.
       */
      vari(const double x): 
        val_(x),
        adj_(0.0) {
        allocator_->var_stack_.push_back(this);
      }

      /**
       * Apply the chain rule to this variable based on the variables
       * on which it depends.  The base implementation in this class
       * is a no-op. 
       */
      virtual void chain() {
      }

      /**
       * Allocate memory from the underlying memory pool.  This memory is
       * is managed by the gradient program and will be recovered as a whole.
       * Classes should not be allocated with this operator if they have
       * non-trivial destructors.
       *
       * @param nbytes Number of bytes to allocate.
       * @return Pointer to allocated bytes.
       */
      static inline void* operator new(size_t nbytes) {
        if (allocator_ == 0)
          allocator_ = new var_allocator();
        return allocator_->alloc(nbytes);
      }

      /**
       * Recover memory used for all variables for reuse.
       */
      static void recover_memory() {
        return allocator_->recover();
        // allocator_.var_stack_.resize(0);
        // allocator_.memalloc_.recover_all();
      }

      /**
       * Return all memory used for gradients back to the system.
       */
      static void free_memory() {
        allocator_->free();
        // allocator_.memalloc_.free_all();
      }

    private:
      /**
       * Compute the gradient for all variables starting from
       * the specified root variable implementation.  Does
       * not recover memory.
       *
       * @param vi Variable implementation for root of partial
       * derivative propagation.
       */
      static void grad(vari* vi) {
        std::vector<vari*>::iterator it = allocator_->var_stack_.end();
        std::vector<vari*>::iterator begin = allocator_->var_stack_.begin();
        // skip to root variable
        for (; (it >= begin) && (*it != vi); --it)
          ;
        vi->adj_ = 1.0; // droot/droot = 1
        // propagate derivates for remaining vars
        for (; it >= begin; --it)
          (*it)->chain();
      }

    };

    namespace {

      class op_v_vari : public vari {
      protected:
        vari* avi_;
      public:
        op_v_vari(double f, vari* avi) :
          vari(f),
          avi_(avi) {
        }
      };

      class op_vv_vari : public vari {
      protected:
        vari* avi_;
        vari* bvi_;
      public:
        op_vv_vari(double f, vari* avi, vari* bvi):
          vari(f),
          avi_(avi),
          bvi_(bvi) {
        }
      };

      class op_vd_vari : public vari {
      protected:
        vari* avi_;
        double bd_;
      public:
        op_vd_vari(double f, vari* avi, double b) :
          vari(f),
          avi_(avi),
          bd_(b) {
        }
      };

      class op_dv_vari : public vari {
      protected:
        double ad_;
        vari* bvi_;
      public:
        op_dv_vari(double f, double a, vari* bvi) :
          vari(f),
          ad_(a),
          bvi_(bvi) {
        }
      };

      class op_vvv_vari : public vari {
      protected:
        vari* avi_;
        vari* bvi_;
        vari* cvi_;
      public:
        op_vvv_vari(double f, vari* avi, vari* bvi, vari* cvi) :
          vari(f),
          avi_(avi),
          bvi_(bvi),
          cvi_(cvi) {
        }
      };

      class op_vvd_vari : public vari {
      protected:
        vari* avi_;
        vari* bvi_;
        double cd_;
      public:
        op_vvd_vari(double f, vari* avi, vari* bvi, double c) :
          vari(f),
          avi_(avi),
          bvi_(bvi),
          cd_(c) {
        }
      };

      class op_vdv_vari : public vari {
      protected:
        vari* avi_;
        double bd_;
        vari* cvi_;
      public:
        op_vdv_vari(double f, vari* avi, double b, vari* cvi) :
          vari(f),
          avi_(avi),
          bd_(b), 
          cvi_(cvi) {
        }
      };

      class op_vdd_vari : public vari {
      protected:
        vari* avi_;
        double bd_;
        double cd_;
      public:
        op_vdd_vari(double f, vari* avi, double b, double c) :
          vari(f),
          avi_(avi),
          bd_(b), 
          cd_(c) {
        }
      };

      class op_dvv_vari : public vari {
      protected:
        double ad_;
        vari* bvi_;
        vari* cvi_;
      public:
        op_dvv_vari(double f, double a, vari* bvi, vari* cvi) :
          vari(f),
          ad_(a),
          bvi_(bvi),
          cvi_(cvi) {
        }
      };

      class op_dvd_vari : public vari {
      protected:
        double ad_;
        vari* bvi_;
        double cd_;
      public:
        op_dvd_vari(double f, double a, vari* bvi, double c) :
          vari(f),
          ad_(a),
          bvi_(bvi),
          cd_(c) {
        }
      };

      class op_ddv_vari : public vari {
      protected:
        double ad_;
        double bd_;
        vari* cvi_;
      public:
        op_ddv_vari(double f, double a, double b, vari* cvi) :
          vari(f),
          ad_(a),
          bd_(b),
          cvi_(cvi) {
        }
      };

      class neg_vari : public op_v_vari {
      public: 
        neg_vari(vari* avi) :
          op_v_vari(-(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ -= adj_;
        }
      };


      class add_vv_vari : public op_vv_vari {
      public:
        add_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ + bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ += adj_;
        }
      };

      class add_vd_vari : public op_vd_vari {
      public:
        add_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ + b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class increment_vari : public op_v_vari {
      public:
        increment_vari(vari* avi) :
          op_v_vari(avi->val_ + 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class decrement_vari : public op_v_vari {
      public:
        decrement_vari(vari* avi) :
          op_v_vari(avi->val_ - 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class subtract_vv_vari : public op_vv_vari {
      public:
        subtract_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ - bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ -= adj_;
        }
      };
    
      class subtract_vd_vari : public op_vd_vari {
      public:
        subtract_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ - b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class subtract_dv_vari : public op_dv_vari {
      public:
        subtract_dv_vari(double a, vari* bvi) :
          op_dv_vari(a - bvi->val_, a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_;
        }
      };

      class multiply_vv_vari : public op_vv_vari {
      public:
        multiply_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ * bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += bvi_->val_ * adj_;
          bvi_->adj_ += avi_->val_ * adj_;
        }
      };

      class multiply_vd_vari : public op_vd_vari {
      public:
        multiply_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ * b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
        }
      };

      // (a/b)' = a' * (1 / b) - b' * (a / [b * b])
      class divide_vv_vari : public op_vv_vari {
      public:
        divide_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ / bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ / bvi_->val_;
          bvi_->adj_ -= adj_ * avi_->val_ / (bvi_->val_ * bvi_->val_);
        }
      };

      class divide_vd_vari : public op_vd_vari {
      public:
        divide_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ / b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ / bd_;
        }
      };

      class divide_dv_vari : public op_dv_vari {
      public:
        divide_dv_vari(double a, vari* bvi) :
          op_dv_vari(a / bvi->val_, a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_ * ad_ / (bvi_->val_ * bvi_->val_);
        }
      };

      class exp_vari : public op_v_vari {
      public:
        exp_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };

      class log_vari : public op_v_vari {
      public:
        log_vari(vari* avi) :
          op_v_vari(std::log(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / avi_->val_;
        }
      };

      double LOG_10 = std::log(10.0);
    
      class log10_vari : public op_v_vari {
      public:
        const double exp_val_;
        log10_vari(vari* avi) :
          op_v_vari(std::log10(avi->val_),avi),
          exp_val_(avi->val_) {
        }
        void chain() {
          avi_->adj_ += adj_ / (LOG_10 * exp_val_);
        }
      };

      class sqrt_vari : public op_v_vari {
      public:
        sqrt_vari(vari* avi) :
          op_v_vari(std::sqrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (2.0 * val_);
        }
      };

      class pow_vv_vari : public op_vv_vari {
      public:
        pow_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::pow(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          if (avi_->val_ == 0.0) return; // partials zero, avoids /0 & log(0)
          avi_->adj_ += adj_ * bvi_->val_ * val_ / avi_->val_;
          bvi_->adj_ += adj_ * std::log(avi_->val_) * val_;
        }
      };

      class pow_vd_vari : public op_vd_vari {
      public:
        pow_vd_vari(vari* avi, double b) :
          op_vd_vari(std::pow(avi->val_,b),avi,b) {
        }
        void chain() {
          if (avi_->val_ == 0.0) return; // partials zero, avoids /0 & log(0)
          avi_->adj_ += adj_ * bd_ * val_ / avi_->val_;
        }
      };

      class pow_dv_vari : public op_dv_vari {
      public:
        pow_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::pow(a,bvi->val_),a,bvi) {
        }
        void chain() {
          if (ad_ == 0.0) return; // partials zero, avoids /0 & log(0)
          bvi_->adj_ += adj_ * std::log(ad_) * val_;
        }
      };

      class cos_vari : public op_v_vari {
      public:
        cos_vari(vari* avi) :
          op_v_vari(std::cos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * std::sin(avi_->val_);
        }
      };

      class sin_vari : public op_v_vari {
      public:
        sin_vari(vari* avi) :
          op_v_vari(std::sin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cos(avi_->val_);
        }
      };

      class tan_vari : public op_v_vari {
      public:
        tan_vari(vari* avi) :
          op_v_vari(std::tan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * (1.0 + val_ * val_); 
        }
      };

      class acos_vari : public op_v_vari {
      public:
        acos_vari(vari* avi) :
          op_v_vari(std::acos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };

      class asin_vari : public op_v_vari {
      public:
        asin_vari(vari* avi) :
          op_v_vari(std::asin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };

      class atan_vari : public op_v_vari {
      public:
        atan_vari(vari* avi) :
          op_v_vari(std::atan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 + (avi_->val_ * avi_->val_));
        }
      };

      class atan2_vv_vari : public op_vv_vari {
      public:
        atan2_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::atan2(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bvi_->val_ * bvi_->val_);
          avi_->adj_ += bvi_->val_ / a_sq_plus_b_sq;
          bvi_->adj_ -= avi_->val_ / a_sq_plus_b_sq;
        }
      };

      class atan2_vd_vari : public op_vd_vari {
      public:
        atan2_vd_vari(vari* avi, double b) :
          op_vd_vari(std::atan2(avi->val_,b),avi,b) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bd_ * bd_);
          avi_->adj_ += bd_ / a_sq_plus_b_sq;
        }
      };

      class atan2_dv_vari : public op_dv_vari {
      public:
        atan2_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::atan2(a,bvi->val_),a,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (ad_ * ad_) + (bvi_->val_ * bvi_->val_);
          bvi_->adj_ -= ad_ / a_sq_plus_b_sq;
        }
      };

      class cosh_vari : public op_v_vari {
      public:
        cosh_vari(vari* avi) :
          op_v_vari(std::cosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::sinh(avi_->val_);
        }
      };

      class sinh_vari : public op_v_vari {
      public:
        sinh_vari(vari* avi) :
          op_v_vari(std::sinh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cosh(avi_->val_);
        }
      };

      class tanh_vari : public op_v_vari {
      public:
        tanh_vari(vari* avi) :
          op_v_vari(std::tanh(avi->val_),avi) {
        }
        void chain() {
          double cosh = std::cosh(avi_->val_);
          avi_->adj_ += adj_ / (cosh * cosh);
        }
      };


      class floor_vari : public vari {
      public:
        floor_vari(vari* avi) :
          vari(std::floor(avi->val_)) {
        }
      };

      class ceil_vari : public vari {
      public:
        ceil_vari(vari* avi) :
          vari(std::ceil(avi->val_)) {
        }
      };

      class fmod_vv_vari : public op_vv_vari {
      public:
        fmod_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::fmod(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ -= adj_ * static_cast<int>(avi_->val_ / bvi_->val_);
        }
      };

      class fmod_vd_vari : public op_v_vari {
      public:
        fmod_vd_vari(vari* avi, double b) :
          op_v_vari(std::fmod(avi->val_,b),avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };
      
      class fmod_dv_vari : public op_dv_vari {
      public:
        fmod_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::fmod(a,bvi->val_),a,bvi) {
        }
        void chain() {
          int d = static_cast<int>(ad_ / bvi_->val_);
          bvi_->adj_ -= adj_ * d;
        }
      };





    }

    // ********************* vari UP, var DOWN ***********************************************************


    /**
     * Independent (input) and dependent (output) variables for gradients.
     *
     * An agrad::var is constructed with a double and used like any
     * other scalar.  Arithmetical functions like negation, addition,
     * and subtraction, as well as a range of mathematical functions
     * like exponentiation and powers are overridden to operate on
     * agrad::var values objects.
     */
    class var {
    public:

      typedef double Scalar;

      /**
       * Pointer to the implementation of this variable.  
       *
       * This value should not be modified, but may be accessed in
       * <code>var</code> operators to construct <code>vari</code>
       * instances.
       */
      vari * vi_;

      /**
       * Construct a variable from a pointer to a variable implementation.
       *
       * @param vi Variable implementation. 
       */
      explicit var(vari* vi) :
        vi_(vi) {
      }

      /**
       * Construct a variable for later assignment.   
       *
       * This is implemented as a no-op, leaving the underlying implementation
       * dangling.  Before an assignment, the behavior is thus undefined just
       * as for a basic double.
       */
      var() :
        vi_(0) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param b Value.
       */
      var(bool b) :
        vi_(new vari(static_cast<double>(b))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param c Value.
       */
      var(char c) :
        vi_(new vari(static_cast<double>(c))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(short n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned short n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(long int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned long int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param x Value.
       */
      var(float x) :
        vi_(new vari(static_cast<double>(x))) {
      }

      /**      
       * Construct a variable with the specified value.
       *
       * @param x Value of the variable.
       */
      var(double x) :
        vi_(new vari(x)) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param x Value.
       */
      var(long double x) :
        vi_(new vari(static_cast<double>(x))) {
      }

      /**
       * Return the value of this variable.
       *
       * @return The value of this variable.
       */
      inline double val() const {
        return vi_->val_;
      }

      /**
       * Compute the gradient of this dependent variable with respect to
       * the specified vector of independent variables, assigning the
       * gradients componentwise to the specified vector of gradients.
       *
       * After the computation of the gradient and value, memory is
       * recovered.
       *
       * @param x Vector of independent variables.
       * @param g Gradient vector of partial derivatives of this
       * variable with respect to x.
       */
      void grad(std::vector<var>& x,
                std::vector<double>& g) {
        vari::grad(vi_);
        g.resize(x.size());
        for (size_t i = 0U; i < x.size(); ++i) 
          g[i] = x[i].vi_->adj_;
        vari::recover_memory();
      }

      /**
       * Compute gradients of this dependent variable with respect to
       * all variables on which it depends.  
       *
       * Memory is recovered, but not freed after this operation,
       * calling <code>vari::recover_memory()</code>; see
       * <code>vari::free_all()</code> to release resources back to
       * the system rather than saving them for reuse).
       *
       * Until the next creation of a stan::agrad::var instance, the
       * gradient values will be available from an instance <code>x</code>
       * of <code>stan::agrad::var</code> via <code>x.vi_->adj_</code>.
       * It may be slightly more efficient to do this without the intermediate
       * creation and population of two vectors as done in the two-argument
       * form <code>grad(std::vector<var>&, std::vector<double>&)</code>.
       */
      void grad() {
        vari::grad(vi_);
        vari::recover_memory();
      }

      // COMPOUND ASSIGNMENT OPERATORS
    
      /**
       * The compound add/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a += b) behaves exactly the same way as (a = a + b),
       * creating an intermediate variable representing (a + b).
       * 
       * @param b The variable to add to this variable.
       * @return The result of adding the specified variable to this variable.
       */
      inline var& operator+=(const var& b) {
        vi_ = new add_vv_vari(vi_,b.vi_);
        return *this;
      }

      /**
       * The compound add/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a += b) behaves exactly the same way as (a = a + b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to add to this variable.
       * @return The result of adding the specified variable to this variable.
       */
      inline var& operator+=(const double& b) {
        vi_ = new add_vd_vari(vi_,b);
        return *this;
      }

      /**
       * The compound subtract/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a -= b) behaves exactly the same way as (a = a - b).
       * Note that the result is an assignable lvalue.
       *
       * @param b The variable to subtract from this variable.
       * @return The result of subtracting the specified variable from
       * this variable.
       */
      inline var& operator-=(const var& b) {
        vi_ = new subtract_vv_vari(vi_,b.vi_);
        return *this;
      }

      /**
       * The compound subtract/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a -= b) behaves exactly the same way as (a = a - b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to subtract from this variable.
       * @return The result of subtracting the specified variable from this
       * variable.
       */
      inline var& operator-=(const double& b) {
        vi_ = new subtract_vd_vari(vi_,b);
        return *this;
      }

      /**
       * The compound multiply/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a *= b) behaves exactly the same way as (a = a * b).
       * Note that the result is an assignable lvalue.
       *
       * @param b The variable to multiply this variable by.
       * @return The result of multiplying this variable by the
       * specified variable.
       */
      inline var& operator*=(const var& b) {
        vi_ = new multiply_vv_vari(vi_,b.vi_);
        return *this;
      }

      /**
       * The compound multiply/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a *= b) behaves exactly the same way as (a = a * b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to multiply this variable by.
       * @return The result of multplying this variable by the specified
       * variable.
       */
      inline var& operator*=(const double& b) {
        vi_ = new multiply_vd_vari(vi_,b);
        return *this;
      }

      /**
       * The compound divide/assignment operator for variables (C++).  If this
       * variable is a and the argument is the variable b, then (a /= b)
       * behaves exactly the same way as (a = a / b).  Note that the
       * result is an assignable lvalue.
       *
       * @param b The variable to divide this variable by.
       * @return The result of dividing this variable by the
       * specified variable.
       */
      inline var& operator/=(const var& b) {
        vi_ = new divide_vv_vari(vi_,b.vi_);
        return *this;
      }

      /**
       * The compound divide/assignment operator for scalars (C++). 
       *
       * If this variable is a and the argument is the scalar b, then
       * (a /= b) behaves exactly the same way as (a = a / b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to divide this variable by.
       * @return The result of dividing this variable by the specified
       * variable.
       */
      inline var& operator/=(const double& b) {
        vi_ = new divide_vd_vari(vi_,b);
        return *this;
      };


    };

    // COMPARISON OPERATORS

    /**
     * Equality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is the same as the
     * second's.
     */
    inline bool operator==(const var& a, const var& b) {
      return a.val() == b.val();
    }

    /**
     * Equality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is the same as the
     * second value.
     */
    inline bool operator==(const var& a, const double& b) {
      return a.val() == b;
    }
  
    /**
     * Equality operator comparing a scalar and a variable's value
     * (C++).
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return True if the variable's value is equal to the scalar.
     */
    inline bool operator==(const double& a, const var& b) {
      return a == b.val();
    }

    /**
     * Inequality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is not the same as the
     * second's.
     */
    inline bool operator!=(const var& a, const var& b) {
      return a.val() != b.val();
    }

    /**
     * Inequality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is not the same as the
     * second value.
     */
    inline bool operator!=(const var& a, const double& b) {
      return a.val() != b;
    }

    /**
     * Inequality operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable. 
     * @return True if the first value is not the same as the
     * second variable's value.
     */
    inline bool operator!=(const double& a, const var& b) {
      return a != b.val();
    }

    /**
     * Less than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than second's.
     */
    inline bool operator<(const var& a, const var& b) {
      return a.val() < b.val();
    }

    /**
     * Less than operator comparing variable's value and a double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than second value.
     */
    inline bool operator<(const var& a, const double& b) {
      return a.val() < b;
    }

    /**
     * Less than operator comparing a double and variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than second variable's value.
     */
    inline bool operator<(const double& a, const var& b) {
      return a < b.val();
    }

    /**
     * Greater than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than second's.
     */
    inline bool operator>(const var& a, const var& b) {
      return a.val() > b.val();
    }

    /**
     * Greater than operator comparing variable's value and double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than second value.
     */
    inline bool operator>(const var& a, const double& b) {
      return a.val() > b;
    }

    /**
     * Greater than operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is greater than second variable's value.
     */
    inline bool operator>(const double& a, const var& b) {
      return a > b.val();
    }

    /**
     * Less than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than or equal to
     * the second's.
     */
    inline bool operator<=(const var& a, const var& b) {
      return a.val() <= b.val();
    }

    /**
     * Less than or equal operator comparing a variable's value and a
     * scalar (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than or equal to
     * the second value.
     */
    inline bool operator<=(const var& a, const double& b) {
      return a.val() <= b;
    }

    /**
     * Less than or equal operator comparing a double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than or equal to the second
     * variable's value.
     */
    inline bool operator<=(const double& a, const var& b) {
      return a <= b.val();
    }

    /**
     * Greater than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than or equal
     * to the second's.
     */
    inline bool operator>=(const var& a, const var& b) {
      return a.val() >= b.val();
    }

    /**
     * Greater than or equal operator comparing variable's value and
     * double (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than or equal
     * to second value.
     */
    inline bool operator>=(const var& a, const double& b) {
      return a.val() >= b;
    }

    /**
     * Greater than or equal operator comparing double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if the first value is greater than or equal to the
     * second variable's value.
     */
    inline bool operator>=(const double& a, const var& b) {
      return a >= b.val();
    }

    // LOGICAL OPERATORS

    /**
     * Prefix logical negation for the value of variables (C++).  The
     * expression (!a) is equivalent to negating the scalar value of
     * the variable a.
     *
     * Note that this is the only logical operator defined for
     * variables.  Overridden logical conjunction (&&) and disjunction
     * (||) operators do not apply the same "short circuit" rules
     * as the built-in logical operators.  
     *
     * @param a Variable to negate.
     * @return True if variable is non-zero.
     */
    inline bool operator!(const var& a) {
      return !a.val();
    }

    // ARITHMETIC OPERATORS

    /**
     * Unary plus operator for variables (C++).  
     *
     * The function simply returns its input, because
     *
     * \f$\frac{d}{dx} +x = \frac{d}{dx} x = 1\f$.
     *
     * The effect of unary plus on a built-in C++ scalar type is
     * integer promotion.  Because variables are all 
     * double-precision floating point already, promotion is
     * not necessary.
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    inline var operator+(const var& a) {
      return a;
    }

    /**
     * Unary negation operator for variables (C++).
     *
     * \f$\frac{d}{dx} -x = -1\f$.
     *
     * @param a Argument variable.
     * @return Negation of variable.
     */
    inline var operator-(const var& a) {
      return var(new neg_vari(a.vi_));
    }

    /**
     * Addition operator for variables (C++).
     *
     * The partial derivatives are defined by 
     *
     * \f$\frac{\partial}{\partial x} (x+y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x+y) = 1\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of adding two variables.
     */
    inline var operator+(const var& a, const var& b) {    
      return var(new add_vv_vari(a.vi_,b.vi_));
    }


    /**
     * Addition operator for variable and scalar (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{dx} (x + c) = 1\f$.
     *
     * @param a First variable operand.
     * @param b Second scalar operand.
     * @return Result of adding variable and scalar.
     */
    inline var operator+(const var& a, const double& b) {
      return var(new add_vd_vari(a.vi_,b));
    }

    /**
     * Addition operator for scalar and variable (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{dy} (c + y) = 1\f$.
     *
     * @param a First scalar operand.
     * @param b Second variable operand.
     * @return Result of adding variable and scalar.
     */
    inline var operator+(const double& a, const var& b) {
      return var(new add_vd_vari(b.vi_,a)); // by symmetry
    }

    /**
     * Subtraction operator for variables (C++).
     *
     * The partial derivatives are defined by 
     *
     * \f$\frac{\partial}{\partial x} (x-y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x-y) = -1\f$.
     * 
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of subtracting the second variable from
     * the first.
     */
    inline var operator-(const var& a, const var& b) {
      return var(new subtract_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Subtraction operator for variable and scalar (C++).
     *
     * The derivative for the variable is
     *
     * \f$\frac{\partial}{\partial x} (x-c) = 1\f$, and
     *
     * @param a First variable operand.
     * @param b Second scalar operand.
     * @return Result of subtracting the scalar from the variable.
     */
    inline var operator-(const var& a, const double& b) {
      return var(new subtract_vd_vari(a.vi_,b));
    }

    /**
     * Subtraction operator for scalar and variable (C++).
     *
     * The derivative for the variable is
     *
     * \f$\frac{\partial}{\partial y} (c-y) = -1\f$, and
     *
     * @param a First scalar operand.
     * @param b Second variable operand.
     * @return Result of sutracting a variable from a scalar.
     */
    inline var operator-(const double& a, const var& b) {
      return var(new subtract_dv_vari(a,b.vi_));
    }

    /**
     * Multiplication operator for two variables (C++).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) = x\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const var& b) {
      return var(new multiply_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Multiplication operator for a variable and a scalar (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial x} (x * c) = c\f$, and
     * 
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const double& b) {
      return var(new multiply_vd_vari(a.vi_,b));
    }

    /**
     * Multiplication operator for a scalar and a variable (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial y} (c * y) = c\f$.
     *
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of multiplying the operands.
     */
    inline var operator*(const double& a, const var& b) {
      return var(new multiply_vd_vari(b.vi_,a)); // by symmetry
    }

    /**
     * Division operator for two variables (C++).
     *
     * The partial derivatives for the variables are
     *
     * \f$\frac{\partial}{\partial x} (x/y) = 1/y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x/y) = -x / y^2\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of dividing the first variable by the
     * second.
     */
    inline var operator/(const var& a, const var& b) {
      return var(new divide_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Division operator for dividing a variable by a scalar (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{\partial}{\partial x} (x/c) = 1/c\f$.
     *
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of dividing the variable by the scalar.
     */
    inline var operator/(const var& a, const double& b) {
      return var(new divide_vd_vari(a.vi_,b));
    }

    /**
     * Division operator for dividing a scalar by a variable (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} (c/y) = -c / y^2\f$.
     * 
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of dividing the scalar by the variable.
     */
    inline var operator/(const double& a, const var& b) {
      return var(new divide_dv_vari(a,b.vi_));
    }

    /**
     * Prefix increment operator for variables (C++).  Following C++,
     * (++a) is defined to behave exactly as (a = a + 1.0) does,
     * but is faster and uses less memory.  In particular, the
     * result is an assignable lvalue.
     *
     * @param a Variable to increment.
     * @return Reference the result of incrementing this input variable.
     */
    inline var& operator++(var& a) {
      a.vi_ = new increment_vari(a.vi_);
      return a;
    }

    /**
     * Postfix increment operator for variables (C++).  
     *
     * Following C++, the expression <code>(a++)</code> i s defined to behave like
     * the sequence of operations
     *
     * <code>var temp = a;  a = a + 1.0;  return temp;</code>
     *
     * @param a Variable to increment.
     * @param dummy Unused dummy variable used to distinguish postfix operator
     * from prefix operator.
     * @return Input variable. 
     */
    inline var operator++(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new increment_vari(a.vi_);
      return temp;
    }

    /**
     * Prefix decrement operator for variables (C++).  
     *
     * Following C++, <code>(--a)</code> is defined to behave exactly as 
     *
     * <code>a = a - 1.0)</code>
     *
     * does, but is faster and uses less memory.  In particular,
     * the result is an assignable lvalue.
     *
     * @param a Variable to decrement.
     * @return Reference the result of decrementing this input variable.
     */
    inline var& operator--(var& a) {
      a.vi_ = new decrement_vari(a.vi_);
      return a;
    }

    /**
     * Postfix decrement operator for variables (C++).  
     * 
     * Following C++, the expression <code>(a--)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>var temp = a;  a = a - 1.0;  return temp;</code>
     *
     * @param a Variable to decrement.
     * @param dummy Unused dummy variable used to distinguish suffix operator
     * from prefix operator.
     * @return Input variable. 
     */
    inline var operator--(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new decrement_vari(a.vi_);
      return temp;
    }

    // CMATH EXP AND LOG
  
    /**
     * Return the exponentiation of the specified variable (cmath).
     *
     * @param a Variable to exponentiate.
     * @return Exponentiated variable.
     */
    inline var exp(const var& a) {
      return var(new exp_vari(a.vi_));
    }

    /**
     * Return the natural log of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \log x = \frac{1}{x}\f$.
     *
     * @param a Variable whose log is taken.
     * @return Natural log of variable.
     */
    inline var log(const var& a) {
      return var(new log_vari(a.vi_));
    }

    /**
     * Return the base 10 log of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \log_{10} x = \frac{1}{x \log 10}\f$.
     * 
     * @param a Variable whose log is taken.
     * @return Base 10 log of variable.
     */
    inline var log10(const var& a) {
      return var(new log10_vari(a.vi_));
    }


    // POWER FUNCTIONS

    /**
     * Return the square root of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sqrt{x} = \frac{1}{2 \sqrt{x}}\f$.
     * 
     * @param a Variable whose square root is taken.
     * @return Square root of variable.
     */
    inline var sqrt(const var& a) {
      return var(new sqrt_vari(a.vi_));
    }

    /**
     * Return the base raised to the power of the exponent (cmath).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} \mbox{pow}(x,y) = y x^{y-1}\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{pow}(x,y) = x^y \ \log x\f$.
     *
     * @param base Base variable.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const var& exponent) {
      return var(new pow_vv_vari(base.vi_,exponent.vi_));
    }
  
    /**
     * Return the base variable raised to the power of the exponent
     * scalar (cmath).
     *
     * The derivative for the variable is
     *
     * \f$\frac{d}{dx} \mbox{pow}(x,c) = c x^{c-1}\f$.
     *
     * @param base Base variable.
     * @param exponent Exponent scalar.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const double& exponent) {
      return var(new pow_vd_vari(base.vi_,exponent));
    }

    /**
     * Return the base scalar raised to the power of the exponent
     * variable (cmath).
     *
     * The derivative for the variable is
     * 
     * \f$\frac{d}{d y} \mbox{pow}(c,y) = c^y \log c \f$.
     * 
     * @param base Base scalar.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const double& base, const var& exponent) {
      return var(new pow_dv_vari(base,exponent.vi_));
    }


    // TRIG FUNCTIONS

    /**
     * Return the cosine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cos x = - \sin x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Cosine of variable. 
     */
    inline var cos(const var& a) {
      return var(new cos_vari(a.vi_));
    }

    /**
     * Return the sine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sin x = \cos x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Sine of variable. 
     */
    inline var sin(const var& a) {
      return var(new sin_vari(a.vi_));
    }

    /**
     * Return the tangent of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tan x = \sec^2 x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Tangent of variable. 
     */
    inline var tan(const var& a) {
      return var(new tan_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc cosine of a variable,
     * in radians (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arccos x = \frac{-1}{\sqrt{1 - x^2}}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc cosine of variable, in radians. 
     */
    inline var acos(const var& a) {
      return var(new acos_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc sine, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arcsin x = \frac{1}{\sqrt{1 - x^2}}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc sine of variable, in radians. 
     */
    inline var asin(const var& a) {
      return var(new asin_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc tangent of variable, in radians. 
     */
    inline var atan(const var& a) {
      return var(new atan_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second (cmath).
     *
     *
     * The partial derivatives are defined by
     *
     * $\f$ \frac{\partial}{\partial x} \arctan \frac{x}{y} = \frac{y}{x^2 + y^2}\f$, and
     * 
     * $\f$ \frac{\partial}{\partial y} \arctan \frac{x}{y} = \frac{-x}{x^2 + y^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const var& b) {
      return var(new atan2_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * $\f$ \frac{d}{d x} \arctan \frac{x}{c} = \frac{c}{x^2 + c^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator scalar.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const double& b) {
      return var(new atan2_vd_vari(a.vi_,b));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first scalar divided by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * $\f$ \frac{\partial}{\partial y} \arctan \frac{c}{y} = \frac{-c}{c^2 + y^2}\f$.
     *
     * @param a Numerator scalar.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const double& a, const var& b) {
      return var(new atan2_dv_vari(a,b.vi_));
    }

    // HYPERBOLIC FUNCTIONS

    /**
     * Return the hyperbolic cosine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cosh x = \sinh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic cosine of variable.
     */
    inline var cosh(const var& a) {
      return var(new cosh_vari(a.vi_));
    }

    /**
     * Return the hyperbolic sine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sinh x = \cosh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic sine of variable.
     */
    inline var sinh(const var& a) {
      return var(new sinh_vari(a.vi_));
    }

    /**
     * Return the hyperbolic tangent of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tanh x = \frac{1}{\cosh^2 x}\f$.
     *
     * @param a Variable.
     * @return Hyperbolic tangent of variable.
     */
    inline var tanh(const var& a) {
      return var(new tanh_vari(a.vi_));
    }
  

    // ROUNDING FUNCTIONS

    /**
     * Return the absolute value of the variable (cmath).  
     *
     * Choosing an arbitrary value at the non-differentiable point 0,
     * 
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * where \f$\mbox{sgn}(x)\f$ is the signum function, taking values
     * -1 if \f$x < 0\f$, 0 if \f$x == 0\f$, and 1 if \f$x == 1\f$.
     *
     * The function <code>abs()</code> provides the same behavior, with
     * <code>abs()</code> defined in stdlib.h and <code>fabs()</code> defined in <code>cmath</code>.
     *
     * @param a Input variable.
     * @return Absolute value of variable.
     */
    inline var fabs(const var& a) {
      // cut-and-paste from abs()
      if (a.val() > 0.0)
        return a;
      if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      return var(new vari(0.0));
    }

    /**
     * Return the floor of the specified variable (cmath).  
     *
     * The derivative of the floor function is defined and
     * zero everywhere but at integers, so we set these derivatives
     * to zero for convenience, 
     *
     * \f$\frac{d}{dx} {\lfloor x \rfloor} = 0\f$.
     *
     * The floor function rounds down.  For double values, this is the largest
     * integral value that is not greater than the specified value.
     * Although this function is not differentiable because it is
     * discontinuous at integral values, its gradient is returned as
     * zero everywhere.
     * 
     * @param a Input variable.
     * @return Floor of the variable.
     */
    inline var floor(const var& a) {
      return var(new floor_vari(a.vi_));
    }

    /**
     * Return the ceiling of the specified variable (cmath).
     *
     * The derivative of the ceiling function is defined and
     * zero everywhere but at integers, and we set them to zero for
     * convenience, 
     *
     * \f$\frac{d}{dx} {\lceil x \rceil} = 0\f$.
     *
     * The ceiling function rounds up.  For double values, this is the
     * smallest integral value that is not less than the specified
     * value.  Although this function is not differentiable because it
     * is discontinuous at integral values, its gradient is returned
     * as zero everywhere.
     * 
     * @param a Input variable.
     * @return Ceiling of the variable.
     */
    inline var ceil(const var& a) {
      return var(new ceil_vari(a.vi_));
    }

    /**
     * Return the floating point remainder after dividing the
     * first variable by the second (cmath).
     *
     * The partial derivatives with respect to the variables are defined
     * everywhere but where \f$x = y\f$, but we set these to match other values,
     * with
     *
     * \f$\frac{\partial}{\partial x} \mbox{fmod}(x,y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{fmod}(x,y) = -\lfloor \frac{x}{y} \rfloor\f$.
     *
     * @param a First variable.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing the first variable
     * by the second.
     */
    inline var fmod(const var& a, const var& b) {
      return var(new fmod_vv_vari(a.vi_,b.vi_));
    }
  
    /**
     * Return the floating point remainder after dividing the
     * the first variable by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d x} \mbox{fmod}(x,c) = \frac{1}{c}\f$.
     *
     * @param a First variable.
     * @param b Second scalar.
     * @return Floating pointer remainder of dividing the first variable by
     * the second scalar.
     */
    inline var fmod(const var& a, const double& b) {
      return var(new fmod_vd_vari(a.vi_,b));
    }

    /**
     * Return the floating point remainder after dividing the
     * first scalar by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} \mbox{fmod}(c,y) = -\lfloor \frac{c}{y} \rfloor\f$.
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing first scalar by
     * the second variable.
     */
    inline var fmod(const double& a, const var& b) {
      return var(new fmod_dv_vari(a,b.vi_));
    }


    // STD LIB FUNCTIONS

    /**
     * Return the absolute value of the variable (std).  
     *
     * The value at the undifferentiable point 0 is conveniently set
     * 0, so that
     *
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * The function fabs() provides identical behavior, with
     * abs() defined in stdlib.h and fabs() defined in cmath.
     *
     * @param a Variable input.
     * @return Absolute value of variable.
     */
    inline var abs(const var& a) {   
      // cut-and-paste from fabs()
      if (a.val() > 0.0)
        return a;
      if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      return var(new vari(0.0));
    }

  }

}

#endif
