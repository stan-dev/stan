#include <stan/model/var_decl.hpp>
#include <test/unit/util.hpp>

TEST(var_decl, one) {
  stan::model::var_decl d("foo", "bar", 2, true, false);

  // test getter values
  EXPECT_EQ("foo", d.name());
  EXPECT_EQ("bar", d.type_name());
  EXPECT_EQ(2, d.array_dims());
  EXPECT_TRUE(d.has_lower_bound());
  EXPECT_FALSE(d.has_upper_bound());

  // test constancy of references
  EXPECT_EQ(&d.name(), &d.name());
  EXPECT_EQ(&d.type_name(), &d.type_name());
}
