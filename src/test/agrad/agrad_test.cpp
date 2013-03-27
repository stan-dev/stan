#include <gtest/gtest.h>
#include <stan/agrad/agrad.hpp>
#include <test/agrad/util.hpp>



TEST(AgradRev,a_div_b) {
  AVAR a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[1]);
}

TEST(AgradRev,a_divide_bd) {
  AVAR a = 6.0;
  double b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
}

TEST(AgradRev,ad_divide_b) {
  double a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[0]);
}

TEST(AgradRev,plus_plus_a) {
  AVAR a(5.0);
  EXPECT_FLOAT_EQ(5.0,a.val());
  AVAR f = ++a;
  EXPECT_FLOAT_EQ(6.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,plus_plus_a_2) {
  AVAR a(5.0);
  EXPECT_FLOAT_EQ(5.0,a.val());
  AVAR f = ++a;
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(6.0,a.val());

  // see next test when created later
  AVEC x = createAVEC(a); 

  ++a;
  EXPECT_FLOAT_EQ(7.0,a.val());
  EXPECT_FLOAT_EQ(6.0,f.val());

  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,plus_plus_a_3) {
  AVAR a(5.0);
  AVAR f = ++a;
  ++a; // reassignment loses connection to f
  AVEC x = createAVEC(a); 
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradRev,a_plus_plus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a++;
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_plus_plus_2) {
  AVAR a(5.0);
  AVAR f = a++;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradRev,minus_minus_a) {
  AVAR a(5.0);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,minus_minus_a_2) {
  AVAR a(5.0);
  AVEC x = createAVEC(a);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_minus_minus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a--;
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_minus_minus_2) {
  AVAR a(5.0);
  AVAR f = a--;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradRev,a_pluseq_b) {
  AVAR a(5.0);
  AVAR b(-1.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a += b);
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(-1.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}

TEST(AgradRev,a_pluseq_bd) {
  AVAR a(5.0);
  double b = -1.0;
  AVEC x = createAVEC(a);
  AVAR f = (a += b);
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_minuseq_b) {
  AVAR a(5.0);
  AVAR b(-1.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a -= b);
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(-1.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
  EXPECT_FLOAT_EQ(-1.0,g[1]);
}

TEST(AgradRev,a_negeq_bd) {
  AVAR a(5.0);
  double b = -1.0;
  AVEC x = createAVEC(a);
  AVAR f = (a -= b);
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(6.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_timeseq_b) {
  AVAR a(5.0);
  AVAR b(-1.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a *= b);
  EXPECT_FLOAT_EQ(-5.0,f.val());
  EXPECT_FLOAT_EQ(-5.0,a.val());
  EXPECT_FLOAT_EQ(-1.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
  EXPECT_FLOAT_EQ(5.0,g[1]);
}

TEST(AgradRev,a_timeseq_bd) {
  AVAR a(5.0);
  double b = -1.0;
  AVEC x = createAVEC(a);
  AVAR f = (a *= b);
  EXPECT_FLOAT_EQ(-5.0,f.val());
  EXPECT_FLOAT_EQ(-5.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
}

TEST(AgradRev,a_divideeq_b) {
  AVAR a(6.0);
  AVAR b(-2.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  EXPECT_FLOAT_EQ(-2.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/((-2.0)*(-2.0)),g[1]);
}

TEST(AgradRev,a_divideeq_bd) {
  AVAR a(6.0);
  double b = -2.0;
  AVEC x = createAVEC(a);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
}

TEST(AgradRev,not_a) {
  AVAR a(6.0);
  EXPECT_EQ(0, !a);
  AVAR b(0.0);
  EXPECT_EQ(1, !b);
}

TEST(AgradRev,exp_a) {
  AVAR a(6.0);
  AVAR f = exp(a); // mix exp() functs w/o namespace
  EXPECT_FLOAT_EQ(exp(6.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(exp(6.0),g[0]);
}
TEST(AgradRev,a_ostream) {
  AVAR a = 6.0;
  std::ostringstream os;
  
  os << a;
  EXPECT_EQ ("6:0", os.str());

  os.str("");
  a = 10.5;
  os << a;
  EXPECT_EQ ("10.5:0", os.str());
}

TEST(AgradRev,log_a) {
  AVAR a(5.0);
  AVAR f = log(a); 
  EXPECT_FLOAT_EQ(log(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/5.0,g[0]);
}


TEST(AgradRev,log10_a) {
  AVAR a(5.0);
  AVAR f = log10(a); 
  EXPECT_FLOAT_EQ(log10(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(log(10.0) * 5.0),g[0]);
}

TEST(AgradRev,sqrt_a) {
  AVAR a(5.0);
  AVAR f = sqrt(a); 
  EXPECT_FLOAT_EQ(sqrt(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ((1.0/2.0) * pow(5.0,-0.5), g[0]);
}

TEST(AgradRev,pow_var_var) {
  AVAR a(3.0);
  AVAR b(4.0);
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0 * pow(3.0,4.0-1.0), g[0]);
  EXPECT_FLOAT_EQ(log(3.0) * pow(3.0,4.0), g[1]);
}

TEST(AgradRev,pow_var_double) {
  AVAR a(3.0);
  double b = 4.0;
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0 * pow(3.0,4.0-1.0), g[0]);
}


TEST(AgradRev,pow_double_var) {
  double a = 3.0;
  AVAR b(4.0);
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(log(3.0) * pow(3.0,4.0), g[0]);
}

TEST(AgradRev,cos_var) {
  AVAR a = 0.43;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ(cos(0.43), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(0.43),g[0]);
}

TEST(AgradRev,sin_var) {
  AVAR a = 0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ(sin(0.49), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
   EXPECT_FLOAT_EQ(cos(0.49),g[0]);
}

TEST(AgradRev,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(tan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}

TEST(AgradRev,acos_var) {
  AVAR a = 0.68;
  AVAR f = acos(a);
  EXPECT_FLOAT_EQ(acos(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradRev,asin_var) {
  AVAR a = 0.68;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ(asin(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradRev,atan_var) {
  AVAR a = 0.68;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ(atan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 + (0.68 * 0.68)), g[0]);
}

TEST(AgradRev,atan2_var_var) {
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[1]);
}

TEST(AgradRev,atan2_dvd) {
  AVAR sigma = 1;
  AVEC x = createAVEC(sigma);
  AVAR f = atan2(1.0,sigma) / 3.14;
  VEC g;
  f.grad(x,g);

  AVAR sigma1 = 1;
  AVEC x1 = createAVEC(sigma1);
  AVAR f1 = atan2(1.0,sigma1);
  VEC g1;
  f1.grad(x1,g1);

  EXPECT_FLOAT_EQ(3.14 * g[0],g1[0]);
}
TEST(AgradRev,atan2_var_var__integration) {
  double c = 5.0;
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b) * c;
  EXPECT_FLOAT_EQ(atan2(1.2,3.9)*c,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[1]);
}


TEST(AgradRev,atan2_var_double) {
  AVAR a = 1.2;

  double b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradRev,atan2_double_var) {
  double a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradRev,cosh_var) {
  AVAR a = 0.68;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(cosh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(sinh(0.68), g[0]);
}

TEST(AgradRev,sinh_var) {
  AVAR a = 0.68;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(sinh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cosh(0.68),g[0]);
}

TEST(AgradRev,tanh_var) {
  AVAR a = 0.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(tanh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(0.68) * cosh(0.68)), g[0]);
}

TEST(AgradRev,fabs_var) {
  AVAR a = 0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradRev,fabs_var_2) {
  AVAR a = -0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradRev,fabs_var_3) {
  AVAR a = 0.0;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}


TEST(AgradRev,abs_var) {
  AVAR a = 0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradRev,abs_var_2) {
  AVAR a = -0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradRev,abs_var_3) {
  AVAR a = 0.0;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,floor_var) {
  AVAR a = 1.2;
  AVAR f = floor(a);
  EXPECT_FLOAT_EQ(1.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,fmod_var_var) {
  AVAR a = 2.7;
  AVAR b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(std::fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
  EXPECT_FLOAT_EQ(-2.0,g[1]); // (int)(2.7/1.3) = 2
}

TEST(AgradRev,fmod_var_double) {
  AVAR a = 2.7;
  double b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,fmod_double_var) {
  double a = 2.7;
  AVAR b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-2.0,g[0]); // (int)(2.7/1.3) = 2
}

TEST(AgradRev,jacobian) {
  AVAR x1 = 2.0;
  AVAR x2 = 3.0;
  
  AVAR y1 = x1 * x2;
  AVAR y2 = x1 + x2;
  AVAR y3 = 17.0 * x1;

  AVEC x = createAVEC(x1,x2);
  AVEC y = createAVEC(y1,y2,y3);

  std::vector<std::vector<double> > J;
  jacobian(y,x,J);

  EXPECT_FLOAT_EQ(3.0,J[0][0]); // dy1/dx1
  EXPECT_FLOAT_EQ(2.0,J[0][1]); // dy1/dx2

  EXPECT_FLOAT_EQ(1.0,J[1][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(1.0,J[1][1]); // dy2/dx2

  EXPECT_FLOAT_EQ(17.0,J[2][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(0.0,J[2][1]); // dy2/dx2
}

TEST(AgradRev,free_memory) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
  stan::agrad::free_memory();

  AVAR aa = 2.0;
  AVAR bb = -3.0;
  AVAR ff = aa * bb;
  EXPECT_FLOAT_EQ(-6.0,ff.val());

  AVEC xx = createAVEC(aa,bb);
  VEC grad_ff;
  ff.grad(xx,grad_ff);
  EXPECT_FLOAT_EQ(-3.0,grad_ff[0]);
  EXPECT_FLOAT_EQ(2.0,grad_ff[1]);
}

TEST(AgradRev, smart_ptrs) {
  AVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0, (*a).val_);
  EXPECT_FLOAT_EQ(2.0, a->val_);

  EXPECT_FLOAT_EQ(2.0,(*a.vi_).val_);
  EXPECT_FLOAT_EQ(2.0,a.vi_->val_);
}

TEST(AgradRev, multiple_grads) {
  for (int i = 0; i < 100; ++i) {
    AVAR a = 2.0;
    AVAR b = 3.0 * a;
    AVAR c = sin(a) * b;
    c = c; // fixes warning regarding unused variable
    
    AVAR nothing;
  }
  
  AVAR d = 2.0;
  AVAR e = 3.0;
  AVAR f = d * e;
  
  AVEC x = createAVEC(d,e);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(3.0, d.adj());
  EXPECT_FLOAT_EQ(2.0, e.adj());

  EXPECT_FLOAT_EQ(3.0, grad_f[0]);
  EXPECT_FLOAT_EQ(2.0, grad_f[1]);
}

TEST(AgradRev, stackAllocation) {
  using stan::agrad::vari;
  using stan::agrad::var;

  vari ai(1.0);
  vari bi(2.0);

  var a(&ai);
  var b(&bi);

  AVEC x = createAVEC(a,b);
  var f = a * b;

  VEC g;
  f.grad(x,g);
  
  EXPECT_EQ(2,g.size());
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}
