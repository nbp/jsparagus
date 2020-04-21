use crate::error::Result;
use crate::parser_tables_generated::Term;

/// This macro is pre-processed by the python grammar processor and generate
/// code out-side the current context.
#[macro_export]
macro_rules! grammar_extension {
    ( $($_:tt)* ) => {};
}

#[derive(Debug)]
pub struct TermValue<Value> {
    pub term: Term,
    pub value: Value,
}

pub trait StackPopN<TV> {
    fn pop(&mut self) -> TV;
}

pub trait ParserTrait<'alloc, Value> {
    fn shift(&mut self, tv: TermValue<Value>) -> Result<'alloc, bool>;
    fn check_before_unwind(&self, n: usize);
    fn unshift(&mut self);
    fn rewind(&mut self, n: usize) {
        for _ in 0..n {
            self.unshift();
        }
    }
    fn pop_1(&mut self) -> TermValue<Value>;
    fn pop_n<'a>(&'a mut self, n: usize) -> Box<dyn StackPopN<TermValue<Value>> + 'a>;
    fn replay(&mut self, tv: TermValue<Value>);
    fn shift_replayed(&mut self, state: usize);
    fn epsilon(&mut self, state: usize);
    fn top_state(&self) -> usize;
    fn check_not_on_new_line(&mut self, peek: usize) -> Result<'alloc, bool>;
}
