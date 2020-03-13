pub trait StrictMode {
    // The directive prologue should probably be handled as part of the grammar
    // using lookahead restrictions.
    fn in_directive_prologue_mut(&mut self) -> &mut bool;
    fn in_directive_prologue(&self) -> bool;

    /// Implement body depth.
    fn body_depth(&self) -> u32;
    fn set_body_depth(&mut self, d: u32);

    /// Strict mode is a flag which is scoped and cannot be unset. Thus instead
    /// of keeping a stack of boolean flags, we record the current depth, as
    /// well as the depth at which the "use strict" flag is set.
    fn strict_depth(&self) -> Option<u32>;
    fn set_strict_depth(&mut self, depth: Option<u32>);

    fn set_directive_prologe(&mut self) {
        *self.in_directive_prologue_mut() = true
    }
    fn unset_directive_prologe(&mut self) {
        *self.in_directive_prologue_mut() = false
    }

    fn push_strict(&mut self) {
        // Strict mode is inherited from the last scope we were in.
        self.set_body_depth(self.body_depth() + 1);
    }
    fn maybe_set_strict(&mut self) {
        if self.strict_depth() == None && self.in_directive_prologue() {
            self.set_strict_depth(Some(self.body_depth()))
        }
    }
    fn pop_strict(&mut self) {
        let depth = self.body_depth() - 1;
        if let Some(d) = self.strict_depth() {
            if d > depth {
                self.set_strict_depth(None)
            }
        }
        self.set_body_depth(depth);
    }
}

grammar_extension! {
    impl StrictMode for SyntaxParser {};

    let FunctionBody[Yield, Await] = {
        { push_strict() } FunctionStatementList[?Yield, ?Await] { pop_strict() }
    };

    let ModuleBody = {
        { push_strict() }
        { maybe_set_strict() }
        ModuleItemList
        { pop_strict() }
    };

    /// This is the ScriptBody which needs to be scoped.
    let ScriptBody = {
        { push_strict() }
        StatementList[~Yield, ~Await, ~Return]
        { pop_strict() }
    };
}
