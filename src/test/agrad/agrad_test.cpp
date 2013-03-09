#include <gtest/gtest.h>
#include <stan/agrad/agrad.hpp>

typedef stan::agrad::var AVAR;
typedef std::vector<AVAR> AVEC;
typedef std::vector<double> VEC;

AVEC createAVEC(AVAR x) {
  AVEC v;
  v.push_back(x);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2) {
  AVEC v;
  v.push_back(x1);
  v.push_back(x2);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3) {
  AVEC v;
  v.push_back(x1);
  v.push_back(x2);
  v.push_back(x3);
  return v;
}

TEST(AgradAgrad,undefined) {
  AVAR a;
  EXPECT_TRUE(a.is_uninitialized());
  a = 5;
  EXPECT_FALSE(a.is_uninitialized());
}

TEST(AgradAgrad,a_eq_x) {
  AVAR a = 5.0;
  EXPECT_FLOAT_EQ(5.0,a.val());
}

TEST(AgradAgrad,a_of_x) {
  AVAR a(6.0);
  EXPECT_FLOAT_EQ(6.0,a.val());
}

TEST(AgradAgrad,a__a_eq_x) {
  AVAR a;
  a = 7.0;
  EXPECT_FLOAT_EQ(7.0,a.val());
}

TEST(AgradAgrad,a) {
  AVAR a = 5.0;
  AVEC x = createAVEC(a);
  VEC dx;
  a.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}


TEST(AgradAgrad,eq_a) {
  AVAR a = 5.0;
  AVAR f = a;
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}



TEST(AgradAgrad,a_lt_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  AVAR c = 6.0;
  EXPECT_FALSE(b < c);
  EXPECT_FALSE(c < b);
}

TEST(AgradAgrad,a_lt_y) {
  AVAR a = 5.0;
  double y = 6.0;
  EXPECT_TRUE(a < y);
  EXPECT_FALSE(y < a);
  AVAR b = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}

TEST(AgradAgrad,x_lt_b) {
  double x = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(x < b);
  EXPECT_FALSE(b < x);
  double y = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}


TEST(AgradAgrad,a_gt_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  AVAR c = 6.0;
  EXPECT_FALSE(b > c);
  EXPECT_FALSE(c > b);
}

TEST(AgradAgrad,a_gt_y) {
  AVAR a = 6.0;
  double y = 5.0;
  EXPECT_TRUE(a > y);
  EXPECT_FALSE(y > a);
  AVAR c = 6.0;
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(c > a);
}

TEST(AgradAgrad,x_gt_b) {
  double x = 6.0;
  AVAR b = 5.0;
  EXPECT_TRUE(x > b);
  EXPECT_FALSE(b > x);
  double y = 5.0;
  EXPECT_FALSE(b > y);
  EXPECT_FALSE(y > b);
}

TEST(AgradAgrad,a_lte_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(b <= a);
  AVAR c = 6.0;
  EXPECT_TRUE(b <= c);
  EXPECT_TRUE(c <= b);
}

TEST(AgradAgrad,a_lte_y) {
  AVAR a = 5.0;
  double y = 6.0;
  EXPECT_TRUE(a <= y);
  EXPECT_FALSE(y <= a);
  AVAR c = 5.0;
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(c <= a);
}


TEST(AgradAgrad,x_lte_b) {
  double x = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(x <= b);
  EXPECT_FALSE(b <= x);
  double y = 6.0;
  EXPECT_TRUE(b <= y);
  EXPECT_TRUE(y <= b);
}


TEST(AgradAgrad,a_gte_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(a >= b);
  AVAR c = 6.0;
  EXPECT_TRUE(b >= c);
  EXPECT_TRUE(c >= b);
}

TEST(AgradAgrad,a_gte_y) {
  AVAR a = 6.0;
  double y = 5.0;
  EXPECT_TRUE(a >= y);
  EXPECT_FALSE(y >= a);
  AVAR c = 6.0;
  EXPECT_TRUE(a >= c);
  EXPECT_TRUE(c >= a);
}

TEST(AgradAgrad,x_gte_b) {
  double x = 6.0;
  AVAR b = 5.0;
  EXPECT_TRUE(x >= b);
  EXPECT_FALSE(b >= x);
  double y = 5.0;
  EXPECT_TRUE(b >= y);
  EXPECT_TRUE(y >= b);
}

TEST(AgradAgrad,a_eq_b) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);
  AVAR c = 3.0;
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(c == a);
}

TEST(AgradAgrad,x_eq_b) {
  double x = 2.0;
  AVAR b = 2.0;
  EXPECT_TRUE(x == b);
  EXPECT_TRUE(b == x);
  AVAR c = 3.0;
  EXPECT_FALSE(x == c);
  EXPECT_FALSE(c == x);
}

TEST(AgradAgrad,a_eq_y) {
  AVAR a = 2.0;
  double y = 2.0;
  EXPECT_TRUE(a == y);
  EXPECT_TRUE(y == a);
  double z = 3.0;
  EXPECT_FALSE(a == z);
  EXPECT_FALSE(z == a);
}


TEST(AgradAgrad,a_neq_y) {
  AVAR a = 2.0;
  double y = 3.0;
  EXPECT_TRUE(a != y);
  EXPECT_TRUE(y != a);
  double z = 2.0;
  EXPECT_FALSE(a != z);
  EXPECT_FALSE(z != a);
}

TEST(AgradAgrad,pos_a) {
  AVAR a = 5.0;
  AVAR f = +a;
  EXPECT_FLOAT_EQ(5.0,f.val());
  EXPECT_TRUE(a == +a);
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradAgrad,neg_a) {
  AVAR a = 5.0;
  AVAR f = -a;
  EXPECT_FLOAT_EQ(-5.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(-1.0,dx[0]);
}

TEST(AgradAgrad,a_plus_b) {
  AVAR a = 5.0;
  AVAR b = -1.0;
  AVAR f = a + b;
  EXPECT_FLOAT_EQ(4.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(1.0,dx[1]);
}

TEST(AgradAgrad,a_plus_a) {
  AVAR a = 5.0;
  AVAR f = a + a;
  EXPECT_FLOAT_EQ(10.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(2.0,dx[0]);
}

TEST(AgradAgrad,a_plus_neg_b) {
  AVAR a = 5.0;
  AVAR b = -1.0;
  AVAR f = a + -b;
  EXPECT_FLOAT_EQ(6.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(-1.0,dx[1]);
}

TEST(AgradAgrad,a_plus_x) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = a + z;
  EXPECT_FLOAT_EQ(8.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradAgrad,x_plus_a) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = z + a;
  EXPECT_FLOAT_EQ(8.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}



TEST(AgradAgrad,a_minus_b) {
  AVAR a = 5.0;
  AVAR b = 2.0;
  AVAR f = a - b;
  EXPECT_FLOAT_EQ(3.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(-1.0,dx[1]);
}

TEST(AgradAgrad,a_minus_a) {
  AVAR a = 5.0;
  AVAR f = a - a;
  EXPECT_FLOAT_EQ(0.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(0.0,dx[0]);
}

TEST(AgradAgrad,a_minus_x) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = a - z;
  EXPECT_FLOAT_EQ(2.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradAgrad,x_minus_a) {
  AVAR a = 2.0;
  double z = 5.0;
  AVAR f = z - a;
  EXPECT_FLOAT_EQ(3.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(-1.0,dx[0]);
}

TEST(AgradAgrad,a_times_b) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
}

TEST(AgradAgrad,a_times_a) {
  AVAR a = 2.0;
  AVAR f = a * a;
  EXPECT_FLOAT_EQ(4.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(4.0,grad_f[0]);
}

TEST(AgradAgrad,a_times_y) {
  AVAR a = 2.0;
  double y = -3.0;
  AVAR f = a * y;
  EXPECT_FLOAT_EQ(-6.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-3.0,g[0]);
}
 
TEST(AgradAgrad,x_times_b) {
  double x = 2.0;
  AVAR b = -3.0;
  AVAR f = x * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC v = createAVEC(b);
  VEC g;
  f.grad(v,g);
  EXPECT_FLOAT_EQ(2.0,g[0]);
}

TEST(AgradAgrad,a_div_b) {
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

TEST(AgradAgrad,a_divide_bd) {
  AVAR a = 6.0;
  double b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
}

TEST(AgradAgrad,ad_divide_b) {
  double a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[0]);
}

TEST(AgradAgrad,plus_plus_a) {
  AVAR a(5.0);
  EXPECT_FLOAT_EQ(5.0,a.val());
  AVAR f = ++a;
  EXPECT_FLOAT_EQ(6.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,plus_plus_a_2) {
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

TEST(AgradAgrad,plus_plus_a_3) {
  AVAR a(5.0);
  AVAR f = ++a;
  ++a; // reassignment loses connection to f
  AVEC x = createAVEC(a); 
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradAgrad,a_plus_plus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a++;
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,a_plus_plus_2) {
  AVAR a(5.0);
  AVAR f = a++;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradAgrad,minus_minus_a) {
  AVAR a(5.0);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,minus_minus_a_2) {
  AVAR a(5.0);
  AVEC x = createAVEC(a);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,a_minus_minus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a--;
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,a_minus_minus_2) {
  AVAR a(5.0);
  AVAR f = a--;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradAgrad,a_pluseq_b) {
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

TEST(AgradAgrad,a_pluseq_bd) {
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

TEST(AgradAgrad,a_minuseq_b) {
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

TEST(AgradAgrad,a_negeq_bd) {
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

TEST(AgradAgrad,a_timeseq_b) {
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

TEST(AgradAgrad,a_timeseq_bd) {
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

TEST(AgradAgrad,a_divideeq_b) {
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

TEST(AgradAgrad,a_divideeq_bd) {
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

TEST(AgradAgrad,not_a) {
  AVAR a(6.0);
  EXPECT_EQ(0, !a);
  AVAR b(0.0);
  EXPECT_EQ(1, !b);
}

TEST(AgradAgrad,exp_a) {
  AVAR a(6.0);
  AVAR f = exp(a); // mix exp() functs w/o namespace
  EXPECT_FLOAT_EQ(exp(6.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(exp(6.0),g[0]);
}
TEST(AgradAgrad,a_ostream) {
  AVAR a = 6.0;
  std::ostringstream os;
  
  os << a;
  EXPECT_EQ ("6:0", os.str());

  os.str("");
  a = 10.5;
  os << a;
  EXPECT_EQ ("10.5:0", os.str());
}

TEST(AgradAgrad,log_a) {
  AVAR a(5.0);
  AVAR f = log(a); 
  EXPECT_FLOAT_EQ(log(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/5.0,g[0]);
}


TEST(AgradAgrad,log10_a) {
  AVAR a(5.0);
  AVAR f = log10(a); 
  EXPECT_FLOAT_EQ(log10(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(log(10.0) * 5.0),g[0]);
}

TEST(AgradAgrad,sqrt_a) {
  AVAR a(5.0);
  AVAR f = sqrt(a); 
  EXPECT_FLOAT_EQ(sqrt(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ((1.0/2.0) * pow(5.0,-0.5), g[0]);
}

TEST(AgradAgrad,pow_var_var) {
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

TEST(AgradAgrad,pow_var_double) {
  AVAR a(3.0);
  double b = 4.0;
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0 * pow(3.0,4.0-1.0), g[0]);
}


TEST(AgradAgrad,pow_double_var) {
  double a = 3.0;
  AVAR b(4.0);
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(log(3.0) * pow(3.0,4.0), g[0]);
}

TEST(AgradAgrad,cos_var) {
  AVAR a = 0.43;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ(cos(0.43), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(0.43),g[0]);
}

TEST(AgradAgrad,sin_var) {
  AVAR a = 0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ(sin(0.49), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
   EXPECT_FLOAT_EQ(cos(0.49),g[0]);
}

TEST(AgradAgrad,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(tan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}

TEST(AgradAgrad,acos_var) {
  AVAR a = 0.68;
  AVAR f = acos(a);
  EXPECT_FLOAT_EQ(acos(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradAgrad,asin_var) {
  AVAR a = 0.68;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ(asin(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradAgrad,atan_var) {
  AVAR a = 0.68;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ(atan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 + (0.68 * 0.68)), g[0]);
}

TEST(AgradAgrad,atan2_var_var) {
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

TEST(AgradAgrad,atan2_dvd) {
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
TEST(AgradAgrad,atan2_var_var__integration) {
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


TEST(AgradAgrad,atan2_var_double) {
  AVAR a = 1.2;

  double b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradAgrad,atan2_double_var) {
  double a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradAgrad,cosh_var) {
  AVAR a = 0.68;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(cosh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(sinh(0.68), g[0]);
}

TEST(AgradAgrad,sinh_var) {
  AVAR a = 0.68;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(sinh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cosh(0.68),g[0]);
}

TEST(AgradAgrad,tanh_var) {
  AVAR a = 0.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(tanh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(0.68) * cosh(0.68)), g[0]);
}

TEST(AgradAgrad,fabs_var) {
  AVAR a = 0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradAgrad,fabs_var_2) {
  AVAR a = -0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradAgrad,fabs_var_3) {
  AVAR a = 0.0;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}


TEST(AgradAgrad,abs_var) {
  AVAR a = 0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradAgrad,abs_var_2) {
  AVAR a = -0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradAgrad,abs_var_3) {
  AVAR a = 0.0;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradAgrad,floor_var) {
  AVAR a = 1.2;
  AVAR f = floor(a);
  EXPECT_FLOAT_EQ(1.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradAgrad,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradAgrad,fmod_var_var) {
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

TEST(AgradAgrad,fmod_var_double) {
  AVAR a = 2.7;
  double b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradAgrad,fmod_double_var) {
  double a = 2.7;
  AVAR b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-2.0,g[0]); // (int)(2.7/1.3) = 2
}

TEST(AgradAgrad,jacobian) {
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

TEST(AgradAgrad,free_memory) {
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

TEST(AgradAgrad, smart_ptrs) {
  AVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0, (*a).val_);
  EXPECT_FLOAT_EQ(2.0, a->val_);

  EXPECT_FLOAT_EQ(2.0,(*a.vi_).val_);
  EXPECT_FLOAT_EQ(2.0,a.vi_->val_);
}

TEST(AgradAgrad, multiple_grads) {
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

TEST(AgradAgrad, stackAllocation) {
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
