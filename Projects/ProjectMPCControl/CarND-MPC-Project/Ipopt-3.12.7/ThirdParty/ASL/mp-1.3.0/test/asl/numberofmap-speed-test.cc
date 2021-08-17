#include "asl/aslbuilder.h"
#include "mp/clock.h"
#include "mp/nl.h"

struct CreateVar {
  int operator()() { return 0; }
};

int main() {
  mp::asl::internal::ASLBuilder b;
  int num_exprs = 10000;
  mp::ProblemInfo pi = mp::ProblemInfo();
  pi.num_vars = num_exprs;
  pi.num_objs = 1;
  b.SetInfo(pi);
  mp::asl::NumberOfMap<int, CreateVar> map((CreateVar()));
  mp::asl::NumericConstant n = b.MakeNumericConstant(0);
  std::vector<mp::asl::NumberOfExpr> exprs(num_exprs);
  for (int i = 0; i < num_exprs; ++i) {
    mp::asl::NumericExpr args[] = {n, b.MakeVariable(i)};
    exprs[i] = b.MakeNumberOf(args);
  }
  mp::steady_clock::time_point start = mp::steady_clock::now();
  for (int i = 0; i < num_exprs; ++i)
    map.Add(0, exprs[i]);
  mp::steady_clock::time_point end = mp::steady_clock::now();
  fmt::print("Executed NumberOfMap.Add {} times in {} s.\n",
    num_exprs, mp::duration_cast< mp::duration<double> >(end - start).count());
}
