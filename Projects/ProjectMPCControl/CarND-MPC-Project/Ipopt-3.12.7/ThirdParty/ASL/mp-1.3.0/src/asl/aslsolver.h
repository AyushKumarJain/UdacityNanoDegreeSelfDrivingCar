﻿/*
 ASL solver

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

#ifndef MP_ASL_SOLVER_H_
#define MP_ASL_SOLVER_H_

#include "mp/solver.h"
#include "asl/aslbuilder.h"

namespace mp {

class ASLSolver : public SolverImpl<asl::internal::ASLBuilder> {
 private:
  void RegisterSuffixes(ASL *asl);

 class ASLSolutionHandler : public SolutionHandler {
  private:
   SolutionHandler &handler_;
   ASLProblem &problem_;

  public:
   ASLSolutionHandler(SolutionHandler &h, ASLProblem &p)
     : handler_(h), problem_(p) {}

   void HandleFeasibleSolution(fmt::StringRef message,
       const double *values, const double *dual_values, double obj_value) {
     handler_.HandleFeasibleSolution(message, values, dual_values, obj_value);
   }

   void HandleSolution(int status, fmt::StringRef message,
       const double *values, const double *dual_values, double obj_value) {
     problem_.set_solve_code(status);
     handler_.HandleSolution(status, message, values, dual_values, obj_value);
   }
 };

 protected:
  virtual void DoSolve(ASLProblem &p, SolutionHandler &sh) = 0;

 public:
  ASLSolver(fmt::StringRef name, fmt::StringRef long_name = 0,
            long date = 0, int flags = 0);

  typedef asl::internal::ASLHandler NLProblemBuilder;

  ASLProblem::Proxy GetProblemBuilder(fmt::StringRef stub);

  // Solves a problem and report solutions via the solution handler.
  void Solve(ASLProblem &problem, SolutionHandler &sh);

  void Solve(asl::internal::ASLBuilder &builder, SolutionHandler &sh) {
    ASLProblem problem(builder.GetProblem());
    Solve(problem, sh);
  }
};
}

#endif  // MP_ASL_SOLVER_H_
