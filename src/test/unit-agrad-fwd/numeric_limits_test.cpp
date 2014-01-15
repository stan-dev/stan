#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/fwd/numeric_limits.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFwdNumericLimits,All_Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::isnan;
  using stan::math::INFTY;

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::is_specialized);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::is_specialized);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::is_specialized);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::is_specialized);

  EXPECT_FLOAT_EQ(2.22507e-308, 
                  std::numeric_limits<fvar<double> >::min().val_);
  EXPECT_FLOAT_EQ(2.22507e-308, 
                  std::numeric_limits<fvar<var> >::min().val_.val());
  EXPECT_FLOAT_EQ(2.22507e-308, 
                  std::numeric_limits<fvar<fvar<double> > >::min().val_.val_);
  EXPECT_FLOAT_EQ(2.22507e-308, 
               std::numeric_limits<fvar<fvar<var> > >::min().val_.val_.val());

  EXPECT_FLOAT_EQ(1.79769e+308, 
                  std::numeric_limits<fvar<double> >::max().val_);
  EXPECT_FLOAT_EQ(1.79769e+308, 
                  std::numeric_limits<fvar<var> >::max().val_.val());
  EXPECT_FLOAT_EQ(1.79769e+308, 
                  std::numeric_limits<fvar<fvar<double> > >::max().val_.val_);
  EXPECT_FLOAT_EQ(1.79769e+308, 
               std::numeric_limits<fvar<fvar<var> > >::max().val_.val_.val());

  EXPECT_FLOAT_EQ(53, std::numeric_limits<fvar<double> >::digits);
  EXPECT_FLOAT_EQ(53, std::numeric_limits<fvar<var> >::digits);
  EXPECT_FLOAT_EQ(53, std::numeric_limits<fvar<fvar<double> > >::digits);
  EXPECT_FLOAT_EQ(53, std::numeric_limits<fvar<fvar<var> > >::digits);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::is_signed);
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::is_signed);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::is_signed);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::is_signed);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::is_integer);
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::is_integer);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::is_integer);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::is_integer);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::is_exact);
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::is_exact);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::is_exact);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::is_exact);

  EXPECT_FLOAT_EQ(2, std::numeric_limits<fvar<double> >::radix);
  EXPECT_FLOAT_EQ(2, std::numeric_limits<fvar<var> >::radix);
  EXPECT_FLOAT_EQ(2, std::numeric_limits<fvar<fvar<double> > >::radix);
  EXPECT_FLOAT_EQ(2, std::numeric_limits<fvar<fvar<var> > >::radix);

  EXPECT_FLOAT_EQ(2.220446e-16, 
                  std::numeric_limits<fvar<double> >::epsilon().val_);
  EXPECT_FLOAT_EQ(2.220446e-16, 
                  std::numeric_limits<fvar<var> >::epsilon().val_.val());
  EXPECT_FLOAT_EQ(2.220446e-16, 
            std::numeric_limits<fvar<fvar<double> > >::epsilon().val_.val_);
  EXPECT_FLOAT_EQ(2.220446e-16, 
         std::numeric_limits<fvar<fvar<var> > >::epsilon().val_.val_.val());

  EXPECT_FLOAT_EQ(0.5, 
                  std::numeric_limits<fvar<double> >::round_error().val_);
  EXPECT_FLOAT_EQ(0.5, 
                  std::numeric_limits<fvar<var> >::round_error().val_.val());
  EXPECT_FLOAT_EQ(0.5, 
         std::numeric_limits<fvar<fvar<double> > >::round_error().val_.val_);
  EXPECT_FLOAT_EQ(0.5, 
      std::numeric_limits<fvar<fvar<var> > >::round_error().val_.val_.val());

  EXPECT_FLOAT_EQ(-1021, std::numeric_limits<fvar<double> >::min_exponent);
  EXPECT_FLOAT_EQ(-1021, std::numeric_limits<fvar<var> >::min_exponent);
  EXPECT_FLOAT_EQ(-1021, 
                  std::numeric_limits<fvar<fvar<double> > >::min_exponent);
  EXPECT_FLOAT_EQ(-1021, 
                  std::numeric_limits<fvar<fvar<var> > >::min_exponent);

  EXPECT_FLOAT_EQ(-307, std::numeric_limits<fvar<double> >::min_exponent10);
  EXPECT_FLOAT_EQ(-307, std::numeric_limits<fvar<var> >::min_exponent10);
  EXPECT_FLOAT_EQ(-307, 
                  std::numeric_limits<fvar<fvar<double> > >::min_exponent10);
  EXPECT_FLOAT_EQ(-307, 
                  std::numeric_limits<fvar<fvar<var> > >::min_exponent10);

  EXPECT_FLOAT_EQ(1024, std::numeric_limits<fvar<double> >::max_exponent);
  EXPECT_FLOAT_EQ(1024, std::numeric_limits<fvar<var> >::max_exponent);
  EXPECT_FLOAT_EQ(1024, 
                  std::numeric_limits<fvar<fvar<double> > >::max_exponent);
  EXPECT_FLOAT_EQ(1024, 
                  std::numeric_limits<fvar<fvar<var> > >::max_exponent);

  EXPECT_FLOAT_EQ(308, std::numeric_limits<fvar<double> >::max_exponent10);
  EXPECT_FLOAT_EQ(308, std::numeric_limits<fvar<var> >::max_exponent10);
  EXPECT_FLOAT_EQ(308, 
                  std::numeric_limits<fvar<fvar<double> > >::max_exponent10);
  EXPECT_FLOAT_EQ(308, 
                  std::numeric_limits<fvar<fvar<var> > >::max_exponent10);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::has_infinity);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::has_infinity);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::has_infinity);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::has_infinity);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::has_quiet_NaN);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::has_quiet_NaN);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::has_quiet_NaN);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::has_quiet_NaN);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::has_signaling_NaN);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::has_signaling_NaN);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::has_signaling_NaN);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::has_signaling_NaN);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::has_denorm);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::has_denorm);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::has_denorm);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::has_denorm);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::has_denorm_loss);  
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::has_denorm_loss);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::has_denorm_loss);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::has_denorm_loss);

  EXPECT_FLOAT_EQ(INFTY, std::numeric_limits<fvar<double> >::infinity().val_);
  EXPECT_FLOAT_EQ(INFTY, 
                  std::numeric_limits<fvar<var> >::infinity().val_.val());
  EXPECT_FLOAT_EQ(INFTY, 
         std::numeric_limits<fvar<fvar<double> > >::infinity().val_.val_);
  EXPECT_FLOAT_EQ(INFTY, 
      std::numeric_limits<fvar<fvar<var> > >::infinity().val_.val_.val());

  isnan(std::numeric_limits<fvar<double> >::quiet_NaN().val_);
  isnan(std::numeric_limits<fvar<var> >::quiet_NaN().val_.val());
  isnan(std::numeric_limits<fvar<fvar<double> > >::quiet_NaN().val_.val_);
  isnan(std::numeric_limits<fvar<fvar<var> > >::quiet_NaN().val_.val_.val());

  isnan(std::numeric_limits<fvar<double> >::signaling_NaN().val_);
  isnan(std::numeric_limits<fvar<var> >::signaling_NaN().val_.val());
  isnan(std::numeric_limits<fvar<fvar<double> > >::signaling_NaN().val_.val_);
  isnan(std::numeric_limits<fvar<fvar<var> > >::signaling_NaN().val_.val_.val());

  EXPECT_FLOAT_EQ(4.94066e-324, 
                  std::numeric_limits<fvar<double> >::denorm_min().val_);
  EXPECT_FLOAT_EQ(4.94066e-324, 
                  std::numeric_limits<fvar<var> >::denorm_min().val_.val());
  EXPECT_FLOAT_EQ(4.94066e-324, 
         std::numeric_limits<fvar<fvar<double> > >::denorm_min().val_.val_);
  EXPECT_FLOAT_EQ(4.94066e-324, 
      std::numeric_limits<fvar<fvar<var> > >::denorm_min().val_.val_.val());

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::is_iec559);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::is_iec559);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::is_iec559);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::is_iec559);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::is_bounded);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::is_bounded);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::is_bounded);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::is_bounded);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::is_modulo);  
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::is_modulo);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::is_modulo);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::is_modulo);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::traps);  
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::traps);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::traps);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::traps);

  EXPECT_FALSE(std::numeric_limits<fvar<double> >::tinyness_before);  
  EXPECT_FALSE(std::numeric_limits<fvar<var> >::tinyness_before);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<double> > >::tinyness_before);
  EXPECT_FALSE(std::numeric_limits<fvar<fvar<var> > >::tinyness_before);

  EXPECT_TRUE(std::numeric_limits<fvar<double> >::round_style);  
  EXPECT_TRUE(std::numeric_limits<fvar<var> >::round_style);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<double> > >::round_style);
  EXPECT_TRUE(std::numeric_limits<fvar<fvar<var> > >::round_style);
}
