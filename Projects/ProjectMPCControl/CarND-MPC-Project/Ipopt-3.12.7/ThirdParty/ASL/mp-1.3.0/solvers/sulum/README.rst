Sulum
=====

The solver ``sulum`` uses `Sulum <http://www.sulumoptimization.com/>`_
to solve LP or MIP problems.

Normally the ``sulum`` solver is invoked by AMPL's ``solve`` command,
which gives the invocation
::

     sulum stub -AMPL

in which ``stub.nl`` is an AMPL generic output file (possibly written
by ``ampl -obstub`` or ``ampl -ogstub``).  After solving the problem,
the solver writes a ``stub.sol`` file for use by ampl's ``solve`` and
``solution`` commands. When you run ampl, this all happens automatically
if you give the AMPL commands
::

     option solver sulum;
     solve;

You can control the solver by setting the environment variable
``sulum_options`` appropriately (either by using ampl's ``option`` command,
or by using the shell's ``set`` and ``export`` commands before you invoke ampl).
You can put one or more (white-space separated) option assignments in
``$sulum_options``. The option ``version`` doesn't take a value:

=======      ==================================================
Phrase       Meaning
=======      ==================================================
version      Report version details before solving the problem.
=======      ==================================================

Others are name-value pairs separated by '=', as in
::

     simtimelimit=600

which limits the simplex optimizer time to 600 seconds.

The following command prints the full list of options with descriptions:
::

     sulum -=

See `Sulum Options for AMPL <http://ampl.com/products/solvers/sulum-options/>`_
for the full list of options.

solve_result_num values
-----------------------

Here is a table of ``solve_result_num`` values that ``sulum`` can return
to an AMPL session, along with the text that appears in the associated
``solve_message``.

=====   ===============================
Value   Message
=====   ===============================
    0   optimal solution
  100   feasible solution
  101   dual feasible solution
  102   integer feasible solution
  200   infeasible problem
  201   infeasible or unbounded
  202   integer infeasible
  203   integer infeasible or unbounded
  600   interrupted
=====   ===============================

-------------------

If you invoke ``sulum stub -AMPL`` or ``sulum stub``, you can also
supply additional command-line arguments of the form name=value.
Such arguments override specifications in ``$sulum_options``.  Example::

     ampl -obfoo foo.model foo.data
     nohup sulum -s foo 2>>err&

to solve a problem whose solution will take a while; after it finishes,
::

     ampl foo.model foo.data -
     solution foo.sol;
     display ... /* things involving the computed solution */;

(Here, ``-`` denotes standard input, and ampl reads the ``solution...``
and ``display...`` lines.)
