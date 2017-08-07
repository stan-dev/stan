#include <stan/model/sized_var_decl.hpp>
#include <test/unit/util.hpp>

TEST(sized_var_decl, one) {
  using stan::model::sized_var_decl;
  std::vector<int> sizes;
  sizes.push_back(3);
  sizes.push_back(4);
  sized_var_decl d("foo", "bar", 3, sizes);
  EXPECT_EQ("foo", d.name());
  EXPECT_EQ("bar", d.type_name());
  EXPECT_EQ(3, d.array_dims());
  stan::test::expect_eq(sizes, d.sizes());
}
