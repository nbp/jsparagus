use generated_parser::{
    reduce, AstBuilder, ErrorCode, ParseError, Result, StackValue, TerminalId, Token, TABLES,
};

const ACCEPT: i16 = -0x7fff;
const ERROR: i16 = ACCEPT - 1;

#[derive(Clone, Copy, Debug)]
struct Action(i16);

impl Action {
    fn is_shift(self) -> bool {
        0 <= self.0
    }

    fn shift_state(self) -> usize {
        assert!(self.is_shift());
        self.0 as usize
    }

    fn is_reduce(self) -> bool {
        ACCEPT < self.0 && self.0 < 0
    }

    fn reduce_prod_index(self) -> usize {
        assert!(self.is_reduce());
        (-self.0 - 1) as usize
    }

    fn is_accept(self) -> bool {
        self.0 == ACCEPT
    }

    fn is_error(self) -> bool {
        self.0 == ERROR
    }
}

pub struct Parser<'alloc> {
    state_stack: Vec<usize>,
    node_stack: Vec<StackValue<'alloc>>,
    handler: AstBuilder<'alloc>,
}

impl<'alloc> Parser<'alloc> {
    pub fn new(handler: AstBuilder<'alloc>, entry_state: usize) -> Self {
        TABLES.check();
        assert!(entry_state < TABLES.state_count);

        Self {
            state_stack: vec![entry_state],
            node_stack: vec![],
            handler,
        }
    }

    fn state(&self) -> usize {
        *self.state_stack.last().unwrap()
    }

    fn action(&self, t: TerminalId) -> Action {
        self.action_at_state(t, self.state())
    }

    fn action_at_state(&self, t: TerminalId, state: usize) -> Action {
        let t = t as usize;
        debug_assert!(t < TABLES.action_width);
        debug_assert!(state < TABLES.state_count);
        let start = TABLES.action_idx[state] as usize;
        let end = TABLES.action_idx[state + 1] as usize;
        let token_masks_start = t * TABLES.masks_per_token;
        let token_masks_end = token_masks_start + TABLES.masks_per_token;
        let token_masks = &TABLES.token_masks[token_masks_start..token_masks_end];

        //println!("state: {:?} + {:?}", state, t);
        let mut result : i16 = 0;
        for code in &TABLES.action_table[start..end] {
            let code = *code as i32;
            let mut map_to = (code & 0xffff) as i16; // drop mode bits and sign-extend.
            let mode = ((code >> 8) as i16) >> 15; // sign-extend.

            // If the mask index refered by the action table exists also in the
            // token_masks index, then set mask_new to 0xffff and 0 otherwise.
            let mask_bit = code.to_be_bytes()[1] & 0x7f;
            let mask_idx = code.to_be_bytes()[0] as usize;
            let mask = token_masks[mask_idx];
            let mask_set = (mask >> mask_bit) & 1;
            let mask_val = !((mask_set as i16) - 1);

            // mode is a sequence or repeat. A sequence would add the
            // terminal id to the destination state, while repeat would only
            // return the same destination state for all terminal id.
            //
            // Sequence (= 0b1) is useful in case each state is produced
            // one after the other, and generated in the order of the
            // terminal id.
            //
            // Repeat (= 0b0) is useful in case of a reduce state where
            // most tokens will reduce the current stack to a non-terminal,
            // which is most likely the same non-terminal independently of
            // the terminal.
            map_to += mode & (t as i16);

            // This works because token_masks are disjoint patterns. Ensuring
            // that only one will match.
            result += map_to & mask_val;
        }
        if result == 0 {
            result = ERROR;
        }
        //println!("  to state: {:?}", result);
        return Action(result);
    }

    fn reduce_all(&mut self, t: TerminalId) -> Result<'alloc, Action> {
        let tables = TABLES;
        let mut action = self.action(t);
        while action.is_reduce() {
            let prod_index = action.reduce_prod_index();
            let nt = reduce(&self.handler, prod_index, &mut self.node_stack)?;
            debug_assert!((nt as usize) < tables.goto_width);
            debug_assert!(self.state_stack.len() >= self.node_stack.len());
            self.state_stack.truncate(self.node_stack.len());
            let prev_state = *self.state_stack.last().unwrap();
            let state_after =
                tables.goto_table[prev_state * tables.goto_width + nt as usize] as usize;
            debug_assert!(state_after < tables.state_count);
            self.state_stack.push(state_after);
            action = self.action(t);
        }

        debug_assert_eq!(self.state_stack.len(), self.node_stack.len() + 1);
        Ok(action)
    }

    pub fn write_token(&mut self, token: &Token<'alloc>) -> Result<'alloc, ()> {
        // Loop for error-handling. The normal path through this code reaches
        // the `return` statement.
        loop {
            let action = self.reduce_all(token.terminal_id)?;
            if action.is_shift() {
                self.node_stack
                    .push(StackValue::Token(self.handler.alloc(token.clone())));
                self.state_stack.push(action.shift_state());
                return Ok(());
            } else {
                assert!(action.is_error());
                self.try_error_handling(token)?;
            }
        }
    }

    pub fn close(&mut self, position: usize) -> Result<'alloc, StackValue<'alloc>> {
        // Loop for error-handling.
        loop {
            let action = self.reduce_all(TerminalId::End)?;
            if action.is_accept() {
                assert_eq!(self.node_stack.len(), 1);
                return Ok(self.node_stack.pop().unwrap());
            } else {
                assert!(action.is_error());
                self.try_error_handling(&Token::basic_token(TerminalId::End, position))?;
            }
        }
    }

    fn parse_error(t: &Token<'alloc>) -> Result<'alloc, ()> {
        Err(if t.terminal_id == TerminalId::End {
            ParseError::UnexpectedEnd
        } else {
            ParseError::SyntaxError(t.clone())
        })
    }

    fn try_error_handling(&mut self, t: &Token<'alloc>) -> Result<'alloc, ()> {
        // Error recovery version of the code in write_terminal. Differences
        // between this and write_terminal are commented below.
        assert!(t.terminal_id != TerminalId::ErrorToken);

        let action = self.reduce_all(TerminalId::ErrorToken)?;
        if action.is_shift() {
            let state = *self.state_stack.last().unwrap();
            let error_code = TABLES.error_codes[state]
                .as_ref()
                .expect("state that accepts an ErrorToken must have an error_code")
                .clone();

            self.recover(t, error_code, action.shift_state())
        } else {
            // On error, don't attempt error handling again.
            assert!(action.is_error());
            Self::parse_error(t)
        }
    }

    fn recover(
        &mut self,
        t: &Token<'alloc>,
        error_code: ErrorCode,
        next_state: usize,
    ) -> Result<'alloc, ()> {
        match error_code {
            ErrorCode::Asi => {
                if t.saw_newline
                    || t.terminal_id == TerminalId::End
                    || t.terminal_id == TerminalId::RightCurlyBracket
                {
                    // Don't actually push an ErrorToken onto the stack here. Treat the
                    // ErrorToken as having been consumed and move to the recovered
                    // state.
                    *self.state_stack.last_mut().unwrap() = next_state;
                    Ok(())
                } else {
                    Self::parse_error(t)
                }
            }
            ErrorCode::DoWhileAsi => {
                // do-while always succeeds in inserting a semicolon.
                *self.state_stack.last_mut().unwrap() = next_state;
                Ok(())
            }
        }
    }

    pub fn can_accept_terminal(&self, t: TerminalId) -> bool {
        let state = self.simulate(t);
        !self.action_at_state(t, state).is_error()
    }

    fn simulate(&self, t: TerminalId) -> usize {
        let mut sp = self.state_stack.len() - 1;
        let mut state = self.state_stack[sp];
        let mut action = self.action_at_state(t, state);
        while action.is_reduce() {
            let prod_index = action.reduce_prod_index();
            let (num_pops, nt) = TABLES.reduce_simulator[prod_index];
            debug_assert!((nt as usize) < TABLES.goto_width);
            debug_assert!(self.state_stack.len() >= self.node_stack.len());
            sp = sp - num_pops;
            let prev_state = self.state_stack[sp];
            state = TABLES.goto_table[prev_state * TABLES.goto_width + nt as usize] as usize;
            debug_assert!(state < TABLES.state_count);
            sp += 1;
            action = self.action_at_state(t, state);
        }

        debug_assert_eq!(self.state_stack.len(), self.node_stack.len() + 1);
        state
    }

    /// Return true if self.close() would succeed.
    pub fn can_close(&self) -> bool {
        // Easy case: no error, parsing just succeeds.
        if self.can_accept_terminal(TerminalId::End) {
            true
        } else {
            // Hard case: maybe error-handling would succeed?  BUG: Need
            // simulator to simulate reduce_all; for now just give up
            false
        }
    }
}
