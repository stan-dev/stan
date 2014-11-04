good: parsed and code generated without a main, *included* (not
      linked) in some other tests, so putting a model here will
      not generate object code.
      All of these are parsed, code generated without a main,
      and compiled with a dummy mail with -fsyntax-only (so no
      object code).

bad: not parsed; put ill-formed models here to test error messages.

