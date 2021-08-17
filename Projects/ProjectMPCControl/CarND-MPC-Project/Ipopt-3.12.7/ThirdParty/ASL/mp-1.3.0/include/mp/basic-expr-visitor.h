/*
 Basic expression visitor

 Copyright (C) 2014 AMPL Optimization Inc

 Permission to use, copy, modify, and distribute this software and its
 documentation for any purpose and without fee is hereby granted,
 provided that the above copyright notice appear in all copies and that
 both that the copyright notice and this permission notice and warranty
 disclaimer appear in supporting documentation.

 The author and AMPL Optimization Inc disclaim all warranties with
 regard to this software, including all implied warranties of
 merchantability and fitness.  In no event shall the author be liable
 for any special, indirect or consequential damages or any damages
 whatsoever resulting from loss of use, data or profits, whether in an
 action of contract, negligence or other tortious action, arising out
 of or in connection with the use or performance of this software.

 Author: Victor Zverovich
 */

#ifndef MP_BASIC_EXPR_VISITOR_H_
#define MP_BASIC_EXPR_VISITOR_H_

#include "mp/common.h"
#include "mp/error.h"

#define MP_DEFINE_EXPR_TYPES(ExprTypes) \
  typedef typename ExprTypes::Expr Expr; \
  typedef typename ExprTypes::NumericExpr NumericExpr; \
  typedef typename ExprTypes::LogicalExpr LogicalExpr; \
  typedef typename ExprTypes::NumericConstant NumericConstant; \
  typedef typename ExprTypes::Variable Variable; \
  typedef typename ExprTypes::CommonExpr CommonExpr; \
  typedef typename ExprTypes::UnaryExpr UnaryExpr; \
  typedef typename ExprTypes::BinaryExpr BinaryExpr; \
  typedef typename ExprTypes::IfExpr IfExpr; \
  typedef typename ExprTypes::PLTerm PLTerm; \
  typedef typename ExprTypes::CallExpr CallExpr; \
  typedef typename ExprTypes::VarArgExpr VarArgExpr; \
  typedef typename ExprTypes::SumExpr SumExpr; \
  typedef typename ExprTypes::CountExpr CountExpr; \
  typedef typename ExprTypes::NumberOfExpr NumberOfExpr; \
  typedef typename ExprTypes::SymbolicNumberOfExpr SymbolicNumberOfExpr; \
  typedef typename ExprTypes::LogicalConstant LogicalConstant; \
  typedef typename ExprTypes::NotExpr NotExpr; \
  typedef typename ExprTypes::BinaryLogicalExpr BinaryLogicalExpr; \
  typedef typename ExprTypes::RelationalExpr RelationalExpr; \
  typedef typename ExprTypes::LogicalCountExpr LogicalCountExpr; \
  typedef typename ExprTypes::ImplicationExpr ImplicationExpr; \
  typedef typename ExprTypes::IteratedLogicalExpr IteratedLogicalExpr; \
  typedef typename ExprTypes::PairwiseExpr PairwiseExpr; \
  typedef typename ExprTypes::StringLiteral StringLiteral

namespace mp {

// A basic expression visitor that can be used with different expression
// hierarchies described by ExprTypes.
//
// BasicExprVisitor uses the curiously recurring template pattern:
// http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
template <typename Impl, typename Result, typename LResult, typename ExprTypes>
class BasicExprVisitor {
 public:
  MP_DEFINE_EXPR_TYPES(ExprTypes);

  Result Visit(NumericExpr e);
  LResult Visit(LogicalExpr e);

  Result VisitUnhandledNumericExpr(NumericExpr e) {
    throw MakeUnsupportedError(str(e.kind()));
  }

  LResult VisitUnhandledLogicalExpr(LogicalExpr e) {
    throw MakeUnsupportedError(str(e.kind()));
  }

  Result VisitNumericConstant(NumericConstant c) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(c));
  }

  Result VisitVariable(Variable v) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(v));
  }

  // Visits a unary expression or a function taking one argument.
  Result VisitUnary(UnaryExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitMinus(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAbs(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitFloor(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitCeil(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitSqrt(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitPow2(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitExp(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitLog(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitLog10(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitSin(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitSinh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitCos(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitCosh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitTan(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitTanh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAsin(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAsinh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAcos(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAcosh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAtan(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  Result VisitAtanh(UnaryExpr e) {
    return MP_DISPATCH(VisitUnary(e));
  }

  // Visits a binary expression or a function taking two arguments.
  Result VisitBinary(BinaryExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitAdd(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitSub(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitLess(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitMul(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitDiv(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitIntDiv(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitMod(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitPow(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitPowConstBase(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitPowConstExp(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  // Visits a function taking two arguments.
  Result VisitBinaryFunc(BinaryExpr e) {
    return MP_DISPATCH(VisitBinary(e));
  }

  Result VisitAtan2(BinaryExpr e) {
    return MP_DISPATCH(VisitBinaryFunc(e));
  }

  Result VisitPrecision(BinaryExpr e) {
    return MP_DISPATCH(VisitBinaryFunc(e));
  }

  Result VisitRound(BinaryExpr e) {
    return MP_DISPATCH(VisitBinaryFunc(e));
  }

  Result VisitTrunc(BinaryExpr e) {
    return MP_DISPATCH(VisitBinaryFunc(e));
  }

  Result VisitIf(IfExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitPLTerm(PLTerm e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitCall(CallExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitVarArg(VarArgExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitMin(VarArgExpr e) {
    return MP_DISPATCH(VisitVarArg(e));
  }

  Result VisitMax(VarArgExpr e) {
    return MP_DISPATCH(VisitVarArg(e));
  }

  Result VisitSum(SumExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitNumberOf(NumberOfExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitNumberOfSym(SymbolicNumberOfExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  Result VisitCount(CountExpr e) {
    return MP_DISPATCH(VisitUnhandledNumericExpr(e));
  }

  LResult VisitLogicalConstant(LogicalConstant c) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(c));
  }

  LResult VisitNot(NotExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitBinaryLogical(BinaryLogicalExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitOr(BinaryLogicalExpr e) {
    return MP_DISPATCH(VisitBinaryLogical(e));
  }

  LResult VisitAnd(BinaryLogicalExpr e) {
    return MP_DISPATCH(VisitBinaryLogical(e));
  }

  LResult VisitIff(BinaryLogicalExpr e) {
    return MP_DISPATCH(VisitBinaryLogical(e));
  }

  LResult VisitRelational(RelationalExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitLT(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitLE(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitEQ(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitGE(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitGT(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitNE(RelationalExpr e) {
    return MP_DISPATCH(VisitRelational(e));
  }

  LResult VisitLogicalCount(LogicalCountExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitAtLeast(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitAtMost(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitExactly(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitNotAtLeast(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitNotAtMost(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitNotExactly(LogicalCountExpr e) {
    return MP_DISPATCH(VisitLogicalCount(e));
  }

  LResult VisitImplication(ImplicationExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitIteratedLogical(IteratedLogicalExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitExists(IteratedLogicalExpr e) {
    return MP_DISPATCH(VisitIteratedLogical(e));
  }

  LResult VisitForAll(IteratedLogicalExpr e) {
    return MP_DISPATCH(VisitIteratedLogical(e));
  }

  LResult VisitAllDiff(PairwiseExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }

  LResult VisitNotAllDiff(PairwiseExpr e) {
    return MP_DISPATCH(VisitUnhandledLogicalExpr(e));
  }
};

template <typename Impl, typename Result, typename LResult, typename ET>
Result BasicExprVisitor<Impl, Result, LResult, ET>::Visit(NumericExpr e) {
  // All expressions except OPNUMBEROFs and OPIFSYM are supported.
  switch (e.kind()) {
  default:
    MP_ASSERT(false, "invalid numeric expression");
    // Fall through.
  case expr::CONSTANT:
    return MP_DISPATCH(VisitNumericConstant(
                         ET::template UncheckedCast<NumericConstant>(e)));
  case expr::VARIABLE:
    return MP_DISPATCH(VisitVariable(ET::template UncheckedCast<Variable>(e)));

  // Unary expressions.
  case expr::MINUS:
    return MP_DISPATCH(VisitMinus(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ABS:
    return MP_DISPATCH(VisitAbs(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::FLOOR:
    return MP_DISPATCH(VisitFloor(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::CEIL:
    return MP_DISPATCH(VisitCeil(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::SQRT:
    return MP_DISPATCH(VisitSqrt(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::POW2:
    return MP_DISPATCH(VisitPow2(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::EXP:
    return MP_DISPATCH(VisitExp(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::LOG:
    return MP_DISPATCH(VisitLog(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::LOG10:
    return MP_DISPATCH(VisitLog10(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::SIN:
    return MP_DISPATCH(VisitSin(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::SINH:
    return MP_DISPATCH(VisitSinh(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::COS:
    return MP_DISPATCH(VisitCos(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::COSH:
    return MP_DISPATCH(VisitCosh(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::TAN:
    return MP_DISPATCH(VisitTan(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::TANH:
    return MP_DISPATCH(VisitTanh(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ASIN:
    return MP_DISPATCH(VisitAsin(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ASINH:
    return MP_DISPATCH(VisitAsinh(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ACOS:
    return MP_DISPATCH(VisitAcos(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ACOSH:
    return MP_DISPATCH(VisitAcosh(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ATAN:
    return MP_DISPATCH(VisitAtan(ET::template UncheckedCast<UnaryExpr>(e)));
  case expr::ATANH:
    return MP_DISPATCH(VisitAtanh(ET::template UncheckedCast<UnaryExpr>(e)));

  // Binary expressions.
  case expr::ADD:
    return MP_DISPATCH(VisitAdd(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::SUB:
    return MP_DISPATCH(VisitSub(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::LESS:
    return MP_DISPATCH(VisitLess(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::MUL:
    return MP_DISPATCH(VisitMul(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::DIV:
    return MP_DISPATCH(VisitDiv(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::INT_DIV:
    return MP_DISPATCH(VisitIntDiv(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::MOD:
    return MP_DISPATCH(VisitMod(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::POW:
    return MP_DISPATCH(VisitPow(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::POW_CONST_BASE:
    return MP_DISPATCH(VisitPowConstBase(
                         ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::POW_CONST_EXP:
    return MP_DISPATCH(VisitPowConstExp(
                         ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::ATAN2:
    return MP_DISPATCH(VisitAtan2(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::PRECISION:
    return MP_DISPATCH(VisitPrecision(
                         ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::ROUND:
    return MP_DISPATCH(VisitRound(ET::template UncheckedCast<BinaryExpr>(e)));
  case expr::TRUNC:
    return MP_DISPATCH(VisitTrunc(ET::template UncheckedCast<BinaryExpr>(e)));

  case expr::IF:
    return MP_DISPATCH(VisitIf(ET::template UncheckedCast<IfExpr>(e)));
  case expr::PLTERM:
    return MP_DISPATCH(VisitPLTerm(ET::template UncheckedCast<PLTerm>(e)));
  case expr::CALL:
    return MP_DISPATCH(VisitCall(ET::template UncheckedCast<CallExpr>(e)));
  case expr::MIN:
    return MP_DISPATCH(VisitMin(ET::template UncheckedCast<VarArgExpr>(e)));
  case expr::MAX:
    return MP_DISPATCH(VisitMax(ET::template UncheckedCast<VarArgExpr>(e)));
  case expr::SUM:
    return MP_DISPATCH(VisitSum(ET::template UncheckedCast<SumExpr>(e)));
  case expr::NUMBEROF:
    return MP_DISPATCH(VisitNumberOf(
                         ET::template UncheckedCast<NumberOfExpr>(e)));
  case expr::NUMBEROF_SYM:
    return MP_DISPATCH(VisitNumberOfSym(
                         ET::template UncheckedCast<SymbolicNumberOfExpr>(e)));
  case expr::COUNT:
    return MP_DISPATCH(VisitCount(ET::template UncheckedCast<CountExpr>(e)));
  }
}

template <typename Impl, typename Result, typename LResult, typename ET>
LResult BasicExprVisitor<Impl, Result, LResult, ET>::Visit(LogicalExpr e) {
  switch (e.kind()) {
  default:
    MP_ASSERT(false, "invalid logical expression");
    // Fall through.
  case expr::CONSTANT:
    return MP_DISPATCH(VisitLogicalConstant(
                         ET::template UncheckedCast<LogicalConstant>(e)));
  case expr::NOT:
    return MP_DISPATCH(VisitNot(ET::template UncheckedCast<NotExpr>(e)));
  case expr::OR:
    return MP_DISPATCH(VisitOr(
                         ET::template UncheckedCast<BinaryLogicalExpr>(e)));
  case expr::AND:
    return MP_DISPATCH(VisitAnd(
                         ET::template UncheckedCast<BinaryLogicalExpr>(e)));
  case expr::IFF:
    return MP_DISPATCH(VisitIff(
                         ET::template UncheckedCast<BinaryLogicalExpr>(e)));
  case expr::LT:
    return MP_DISPATCH(VisitLT(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::LE:
    return MP_DISPATCH(VisitLE(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::EQ:
    return MP_DISPATCH(VisitEQ(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::GE:
    return MP_DISPATCH(VisitGE(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::GT:
    return MP_DISPATCH(VisitGT(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::NE:
    return MP_DISPATCH(VisitNE(ET::template UncheckedCast<RelationalExpr>(e)));
  case expr::ATLEAST:
    return MP_DISPATCH(VisitAtLeast(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::ATMOST:
    return MP_DISPATCH(VisitAtMost(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::EXACTLY:
    return MP_DISPATCH(VisitExactly(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::NOT_ATLEAST:
    return MP_DISPATCH(VisitNotAtLeast(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::NOT_ATMOST:
    return MP_DISPATCH(VisitNotAtMost(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::NOT_EXACTLY:
    return MP_DISPATCH(VisitNotExactly(
                         ET::template UncheckedCast<LogicalCountExpr>(e)));
  case expr::IMPLICATION:
    return MP_DISPATCH(VisitImplication(
                         ET::template UncheckedCast<ImplicationExpr>(e)));
  case expr::EXISTS:
    return MP_DISPATCH(VisitExists(
                         ET::template UncheckedCast<IteratedLogicalExpr>(e)));
  case expr::FORALL:
    return MP_DISPATCH(VisitForAll(
                         ET::template UncheckedCast<IteratedLogicalExpr>(e)));
  case expr::ALLDIFF:
    return MP_DISPATCH(VisitAllDiff(
                         ET::template UncheckedCast<PairwiseExpr>(e)));
  case expr::NOT_ALLDIFF:
    return MP_DISPATCH(VisitNotAllDiff(
                         ET::template UncheckedCast<PairwiseExpr>(e)));
  }
}
}  // namespace mp

#endif  // MP_BASIC_EXPR_VISITOR_H_
