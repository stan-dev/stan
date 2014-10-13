no-main: parsed and code generated without main, *included* (not
         linked) in some other tests, so just putting a model here
	 will not generate object code

syntax-only: parsed, code generated with dummy main, compiled
             -fsyntax-only (so no object code)

reference: not parsed; put ill-formed models here to test error
           messages.

