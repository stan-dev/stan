#include <stan/model/var_decl.hpp>
#include <test/unit/util.hpp>

TEST(var_decl, one) {
  using stan::model::var_decl;
  var_decl d("foo", "bar", 2, true, false);
  EXPECT_EQ("foo", d.name());
  EXPECT_EQ("bar", d.type_name());
  EXPECT_EQ(2, d.array_dims());
  EXPECT_TRUE(d.has_lower_bound());
  EXPECT_FALSE(d.has_upper_bound());

  var_decl e("foo", "bar", 2);
  EXPECT_EQ("foo", e.name());
  EXPECT_EQ("bar", e.type_name());
  EXPECT_EQ(2, e.array_dims());
  EXPECT_FALSE(e.has_lower_bound());
  EXPECT_FALSE(e.has_upper_bound());
}
