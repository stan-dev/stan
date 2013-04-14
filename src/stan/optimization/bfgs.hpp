#ifndef __STAN__OPTIMIZATION__BFGS_HPP__
#define __STAN__OPTIMIZATION__BFGS_HPP__

#include <stan/model/prob_grad.hpp>

namespace stan {
  namespace optimization {
    namespace {
      template<typename FunctorType, typename Scalar, typename XType>
      int BTLineSearch(FunctorType &func,
                       Scalar &alpha, 
                       XType &x1, Scalar &f1, XType &gradx1,
                       const XType &p,
                       const XType &x0, const Scalar &f0, const XType &gradx0,
                       const Scalar &rho, const Scalar &c,
                       const Scalar &minAlpha)
      {
        const Scalar cdfdp(c*gradx0.dot(p));
        int ret;
        
        while (1) {
          x1 = x0 + alpha*p;
          ret = func(x1,f1);
          if (ret!=0 && f1 <= f0 + alpha*cdfdp)
            break;
          else
            alpha *= rho;
          
          if (alpha < minAlpha)
            return 1;
        }
        func.df(x1,gradx1);
        return 0;
      }
      
      template<typename FunctorType, typename Scalar, typename XType>
      int WolfLSZoom(Scalar &alpha, XType &newX, Scalar &newF, XType &newDF,
                     FunctorType &func,
                     const XType &x, const Scalar &f, const Scalar &dfp,
                     const Scalar &c1dfp, const Scalar &c2dfp, const XType &p,
                     Scalar alo, Scalar aloF, Scalar aloDFp,
                     Scalar ahi, Scalar ahiF, Scalar ahiDFp,
                     const Scalar &min_range)
      {
        Scalar d1, d2, newDFp;
        
        while (1) {
          if (fabs(alo-ahi) < min_range)
            return 1;
          
          // Perform cubic interpolation to determine next point to try
          d1 = aloDFp + ahiDFp - 3*(aloF-ahiF)/(alo-ahi);
          d2 = sqrt(d1*d1 - aloDFp*ahiDFp);
          if (ahi < alo)
            d2 = -d2;
          alpha = ahi - (ahi - alo)*(ahiDFp + d2 - d1)/(ahiDFp - aloDFp + 2*d2);
          if (alpha < std::min(alo,ahi) || alpha > std::max(alo,ahi))
            alpha = 0.5*(alo+ahi);
          
          newX = x + alpha*p;
          func(newX,newF,newDF);
          newDFp = newDF.dot(p);
          if (newF > (f + alpha*c1dfp) || newF >= aloF) {
            ahi = alpha;
            ahiF = newF;
            ahiDFp = newDFp;
          }
          else {
            if (fabs(newDFp) <= -c2dfp)
              break;
            if (newDFp*(ahi-alo) >= 0) {
              ahi = alo;
              ahiF = aloF;
              ahiDFp = aloDFp;
            }
            alo = alpha;
            aloF = newF;
            aloDFp = newDFp;
          }
        }
        return 0;
      }
      
      template<typename FunctorType, typename Scalar, typename XType>
      int WolfeLineSearch(FunctorType &func,
                          Scalar &alpha,
                          XType &x1, Scalar &f1, XType &gradx1,
                          const XType &p,
                          const XType &x0, const Scalar &f0, const XType &gradx0,
                          const Scalar &c1, const Scalar &c2,
                          const Scalar &minAlpha, const Scalar &maxAlpha)
      {
        const Scalar dfp(gradx0.dot(p));
        const Scalar c1dfp(c1*dfp);
        const Scalar c2dfp(c2*dfp);
        
        Scalar alpha0(minAlpha);
        Scalar alpha1(alpha);
        
        Scalar prevF(f0);
        XType prevDF(gradx0);
        Scalar prevDFp(dfp);
        Scalar newDFp;
        
        int retCode = 0, nits = 0, ret;
        
        while (1) {
          x1 = x0 + alpha1*p;
          ret = func(x1,f1,gradx1);
          if (ret!=0) {
            alpha1 = 0.5*(alpha0+alpha1);
            continue;
          }
          newDFp = gradx1.dot(p);
          if ((f1 > f0 + alpha*c1dfp) || (f1 >= prevF && nits > 0)) {
            retCode = WolfLSZoom(alpha, x1, f1, gradx1,
                                 func,
                                 x0, f0, dfp,
                                 c1dfp, c2dfp, p,
                                 alpha0, prevF, prevDFp,
                                 alpha1, f1, newDFp,
                                 1e-16);
            break;
          }
          if (std::abs(newDFp) <= -c2dfp) {
            alpha = alpha1;
            break;
          }
          if (newDFp >= 0) {
            retCode = WolfLSZoom(alpha, x1, f1, gradx1,
                                 func,
                                 x0, f0, dfp,
                                 c1dfp, c2dfp, p,
                                 alpha1, f1, newDFp,
                                 alpha0, prevF, prevDFp,
                                 1e-16);
            break;
          }
          
          alpha0 = alpha1;
          prevF = f1;
          std::swap(prevDF,gradx1);
          prevDFp = newDFp;
          
          alpha1 = std::min(1.25*alpha0,maxAlpha);
          
          nits++;
        }
        return retCode;
      }

      template<typename FunctorType, typename Scalar = double,
               int DimAtCompile = Eigen::Dynamic,
               int LineSearchMethod = 1>
      class BFGSMinimizer {
      public:
        typedef Eigen::Matrix<Scalar,DimAtCompile,1> VectorT;
        typedef Eigen::Matrix<Scalar,DimAtCompile,DimAtCompile> HessianT;
        
      protected:
        FunctorType &_func;
        Scalar _fx0, _fx1, _alpha;
        size_t _itNum;
        HessianT _B;
        VectorT _gradx0, _gradx1, _x1, _x0;
        VectorT _s;
        
      public:
        const Scalar &f0() const { return _fx0; }
        const VectorT &x0() const { return _x0; }
        const VectorT &g0() const { return _gradx0; }
        const Scalar &f1() const { return _fx1; }
        const VectorT &x1() const { return _x1; }
        const VectorT &g1() const { return _gradx1; }
        
        struct BFGSOptions {
          BFGSOptions() {
            maxIts = 10000;
            rho = 0.75;
            c1 = 1e-4;
            c2 = 0.9;
            minStep = 1e-16;
            minGradNorm = 1e-6;
            alpha0 = 1e3;
            minAlpha = 1e-12;
            maxAlpha = 5.0;
          }
          size_t maxIts;
          Scalar rho;
          Scalar c1;
          Scalar c2;
          Scalar minStep;
          Scalar minGradNorm;
          Scalar alpha0;
          Scalar minAlpha;
          Scalar maxAlpha;
        } _opts;
        
        
        BFGSMinimizer(FunctorType &f) : _func(f) { }
        
        void initialize(const VectorT &x0) {
          _x0 = x0;
          _func(_x0,_fx0,_gradx0);
          
          _B.setIdentity(x0.size(),x0.size());
          _B *= _opts.alpha0;
          
          _alpha = 1;
          _itNum = 0;
        }
        
        int step() {
          VectorT sk, yk, Bksk, rk;
          Scalar gradNorm, thetak, skyk, skBksk;
          int retCode;
          
          _itNum++;
          
          Eigen::LDLT< HessianT > ldlt(_B);
          if (ldlt.info() != Eigen::Success || ldlt.isNegative()) {
            _B.setIdentity();
            _B *= _opts.alpha0;
            _s = -_gradx0/_opts.alpha0;
          }
          else {
            _s = -ldlt.solve(_gradx0);
          }
          
          if (_itNum > 1) {
            _alpha = std::min(1.0, 1.01*2*(_fx0 - _fx1)/_gradx0.dot(_s));
          }
          if (LineSearchMethod == 0) {
            retCode = BTLineSearch(_func, _alpha, _x1, _fx1, _gradx1,
                                   _s, _x0, _fx0, _gradx0, _opts.rho, 
                                   _opts.c1, _opts.minAlpha);
          }
          else if (LineSearchMethod == 1) {
            retCode = WolfeLineSearch(_func, _alpha, _x1, _fx1, _gradx1,
                                      _s, _x0, _fx0, _gradx0,
                                      _opts.c1, _opts.c2, 
                                      _opts.minAlpha, _opts.maxAlpha);
          }
          if (retCode) {
            // Line-search failed
            retCode = 10;
            return retCode;
          }
          _x0.swap(_x1);
          std::swap(_fx0,_fx1);
          _gradx0.swap(_gradx1);
          
          gradNorm = _gradx0.squaredNorm();
          if ((_x1 - _x0).array().abs().maxCoeff() <= _opts.minStep) {
            if (gradNorm <= _opts.minGradNorm) {
              retCode = 1;
            }
            else {
              retCode = 2;
            }
          }
          else {
            retCode = 0;
          }
          
          sk.noalias() = _x0 - _x1;
          yk.noalias() = _gradx0 - _gradx1;
          Bksk.noalias() = _B*sk;
          skyk = yk.dot(sk);
          skBksk = sk.dot(Bksk);
          if (skyk >= 0.2*skBksk) {
            // Full update
            thetak = 1;
            rk = yk;
          }
          else {
            // Damped update (Proceedure 18.2)
            thetak = 0.8*skBksk/(skBksk - skyk);
            rk = thetak*yk + (1.0 - thetak)*Bksk;
          }
          _B.noalias() += rk*rk.transpose()/sk.dot(rk) - Bksk*Bksk.transpose()/skBksk;
          
          return retCode;
        }
        
        int minimize(VectorT &x0) {
          int retcode;
          initialize(x0);
          while (!(retcode = step()));
          x0 = _x0;
          return retcode;
        }
      };
    }
    
    class ModelAdaptor {
    private:
      stan::model::prob_grad& _model;
      std::vector<int> _params_i;
      std::ostream* _output_stream;
      std::vector<double> _x, _g;

    public:
      ModelAdaptor(stan::model::prob_grad& model,
                   const std::vector<int>& params_i,
                   std::ostream* output_stream)
      : _model(model), _params_i(params_i), _output_stream(output_stream) {}
                   
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, double &f) {
        _x.resize(x.size());
        for (size_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = -_model.log_prob(_x, _params_i, _output_stream);
        } catch (const std::exception& e) {
          return 1;
        }

        return 0;
      }
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, double &f, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        _x.resize(x.size());
        for (size_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = -_model.grad_log_prob(_x, _params_i, _g, _output_stream);
        } catch (const std::exception& e) {
          return 1;
        }

        g.resize(_g.size());
        for (size_t i = 0; i < _g.size(); i++)
          g[i] = -_g[i];

        return 0;
      }
      int df(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        double f;
        return (*this)(x,f,g);
      }
    };
    
    class BFGSLineSearch {
    private:
      ModelAdaptor _adaptor;
      BFGSMinimizer<ModelAdaptor> _bfgsOpt;
      
    public:
      BFGSLineSearch(stan::model::prob_grad& model,
                     const std::vector<double>& params_r,
                     const std::vector<int>& params_i,
                     std::ostream* output_stream = 0)
      : _adaptor(model,params_i,output_stream),
        _bfgsOpt(_adaptor) {

        Eigen::Matrix<double,Eigen::Dynamic,1> x;
        x.resize(params_r.size());
        for (size_t i = 0; i < params_r.size(); i++)
          x[i] = params_r[i];
        _bfgsOpt.initialize(x);
      }

      double logp() { return -_bfgsOpt.f0(); }
      void grad(std::vector<double>& g) { 
        const BFGSMinimizer<ModelAdaptor>::VectorT &cg(_bfgsOpt.g0());
        g.resize(cg.size());
        for (size_t i = 0; i < cg.size(); i++)
          g[i] = -cg[i];
      }
      void params_r(std::vector<double>& x) {
        const BFGSMinimizer<ModelAdaptor>::VectorT &cx(_bfgsOpt.g0());
        x.resize(cx.size());
        for (size_t i = 0; i < cx.size(); i++)
          x[i] = cx[i];
      }

      double step() {
        _bfgsOpt.step();
        return logp();
      }
    };

  }

}

#endif
