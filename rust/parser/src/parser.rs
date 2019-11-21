use generated_parser::{
    reduce, AstBuilder, DualIndexMap, ErrorCode, NonterminalId, ParseError, Result, StackValue, TerminalId, Token, TABLES,
};

const ACCEPT: i16 = -0x3fff;
const ERROR: i16 = -0x7fff - 1;

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

struct I8x16(core::arch::x86_64::__m128i);
impl std::fmt::Binary for I8x16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res: [i8;16] = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];
        unsafe {
            use core::arch::x86_64::*; // To use SIMD.
            _mm_storeu_si128(&mut res[0] as *mut i8 as *mut __m128i, self.0);
        };
        write!(f, "({:b},{:b},{:b},{:b}, {:b},{:b},{:b},{:b}, {:b},{:b},{:b},{:b}, {:b},{:b},{:b},{:b})",
               res[0], res[1], res[2], res[3],
               res[4], res[5], res[6], res[7],
               res[8], res[9], res[10], res[11],
               res[12], res[13], res[14], res[15],
        )
    }
}
impl std::fmt::LowerHex for I8x16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res: [i8;16] = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];
        unsafe {
            use core::arch::x86_64::*; // To use SIMD.
            _mm_storeu_si128(&mut res[0] as *mut i8 as *mut __m128i, self.0);
        };
        write!(f, "({:x},{:x},{:x},{:x}, {:x},{:x},{:x},{:x}, {:x},{:x},{:x},{:x}, {:x},{:x},{:x},{:x})",
               res[0], res[1], res[2], res[3],
               res[4], res[5], res[6], res[7],
               res[8], res[9], res[10], res[11],
               res[12], res[13], res[14], res[15],
        )
    }
}
impl std::fmt::Display for I8x16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res: [i8;16] = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];
        unsafe {
            use core::arch::x86_64::*; // To use SIMD.
            _mm_storeu_si128(&mut res[0] as *mut i8 as *mut __m128i, self.0);
        };
        write!(f, "({},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{})",
               res[0], res[1], res[2], res[3],
               res[4], res[5], res[6], res[7],
               res[8], res[9], res[10], res[11],
               res[12], res[13], res[14], res[15],
        )
    }
}
// This function attempt to resolve a dual entry lookup over compressed data.
// It is similar to resolving table[entry1][entry2] on a spare table.
//
// It works by having table_map[entry1] and table_match[entry2] which are then
// checked one against the other.
fn dual_map<'a>(dim: DualIndexMap<'a>, e1: usize, e2: usize, default: i16) -> i16 {
    let e1_start = dim.e1_idx[e1] as usize;
    let e1_end = dim.e1_idx[e1 + 1] as usize;
    let e1_slice : &[u8]= &dim.e1_map[e1_start..e1_end];
    let e2_start = e2 * dim.matches_per_e2;
    let e2_end = e2_start + dim.matches_per_e2;
    let e2_slice : &[u8] = &dim.e2_matches[e2_start..e2_end];
    let e2 = e2 as i16;
    let e1_len = (e1_end - e1_start) / 4;
    assert!(e1_len <= 16);

    let index = unsafe {
        use core::arch::x86_64::*; // To use SIMD.
        // matches_per_token = 19, matches_per_nonterminal = 21
        let e2s1 = _mm_lddqu_si128(&e2_slice[0] as *const u8 as *const __m128i); // e2_slices[0..16]
        let e2s2 = _mm_lddqu_si128(&e2_slice[16] as *const u8 as *const __m128i); // e2_slices[16..32]
        // Note: It is useless to remove parts which are out-side e2 slices, as these are read but not addressed.
        //println!("\ne2s1 = {:b}", I8x16(e2s1));
        //println!("e2s2 = {:b}", I8x16(e2s2));

        // load e1 masks offsets, and split it in 2 to match the e2s1 range and the e2s2 range.
        let e1_matchidx = _mm_lddqu_si128(&e1_slice[0] as *const u8 as *const __m128i); // e1_slices[0..16];
        //println!("e1_matchidx = {}", I8x16(e1_matchidx));
        let vsz = _mm_set1_epi8(16);
        // A substraction will set the sign bit, which when used in pshufb will ignore the lookup and
        // return 0.
        let e1_matchidx_hi = _mm_sub_epi8(e1_matchidx, vsz);
        // The substraction kept all the low bits identical. Using a xor swap the sign bit and make it
        // usable with pshufb.
        let sign = _mm_set1_epi8(0x80u8 as i8);
        let e1_matchidx_lo = _mm_xor_si128(e1_matchidx_hi, sign);

        // Compute the lookup of byte matches using the byte indexes.
        // `for all e1: e2_slice[e1.mask_idx]`
        let matchsets_hi = _mm_shuffle_epi8(e2s2, e1_matchidx_hi);
        let matchsets_lo = _mm_shuffle_epi8(e2s1, e1_matchidx_lo);

        // Compute a mask which is non-zero for all low values of match indexes.
        let zero = _mm_setzero_si128();
        let e1_pick_lo = _mm_min_epi8(e1_matchidx_hi, zero);

        // Merge matchbits from low and high indexes.
        let matchsets = _mm_blendv_epi8(matchsets_hi, matchsets_lo, e1_pick_lo);
        //println!("matchsets = {:b}", I8x16(matchsets));

        // c = shuffle(vshift, b) implies c_i = 1 << b_i
        let vshift = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                  0x80u8 as i8, 0x40, 0x20, 0x10, 8, 4, 2, 1);
        // Compute the shifted bits.
        // load e1_slices[e1_len + 0.. e1_len + 16];
        let e1_matchoff = _mm_lddqu_si128(&e1_slice[e1_len] as *const u8 as *const __m128i);
        let e1_match_bits = _mm_shuffle_epi8(vshift, e1_matchoff);
        let match_bit = _mm_and_si128(e1_match_bits, matchsets);
        let match_miss = _mm_cmpeq_epi8(match_bit, zero);
        let match_mask = _mm_andnot_si128(match_miss, sign);
        //println!("e1_matchbits = {:b}", I8x16(e1_match_bits));
        //println!("match_bit = {:b}", I8x16(match_bit));
        //println!("match_mask = {:x}", I8x16(match_mask));

        // There is at most a single bit set, extract the index (starting at 1, to dinstingish the
        // no-bit set case) of the single bit.
        let bit_index_all = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9,
                                         8, 7, 6, 5, 4, 3, 2, 1);

        // Create a bit mask which reject all inputs which are outside the e1_len range.
        let e1_filter = _mm_set1_epi8((e1_len as i8) + 1);
        let e1_mask = _mm_cmplt_epi8(bit_index_all, e1_filter);
        //println!("e1_mask = {:x}", I8x16(e1_mask));
        // Filter out values which are not valid e1 data.
        let bit_index_e1 = _mm_blendv_epi8(zero, bit_index_all, e1_mask);
        // Compute the index of the mask which bit matched with both e1 and e2.
        let match_index = _mm_blendv_epi8(zero, bit_index_e1, match_mask);
        //println!("match_index = {:x}", I8x16(match_index));
        // Extract the only matching index, if any.
        let collect_index = _mm_sad_epu8(match_index, zero);
        let index_lo = _mm_extract_epi8(collect_index, 0);
        let index_hi = _mm_extract_epi8(collect_index, 8);
        (index_lo + index_hi) as usize
    };

    let result = if index == 0 {
        default
    } else {
        // TODO: find a different place to store the seq/rep bit.
        // TODO: do a single load of an i16?
        let hi = e1_slice[e1_len * 2 - 2 + index * 2];
        let lo = e1_slice[e1_len * 2 - 2 + index * 2 + 1];
        let res : i16 = (hi as i16) << 8 | lo as i16;
        let add_e2_mask = res >> 15;
        ((res << 1) >> 1) + (add_e2_mask & e2)
    };
    //println!("e1: {:?}, e2: {:?} --> {:?}", e1, e2, result);

    /*
    let mut result : i16 = 0;
    for code in e1_slice {
        let code = *code as i32;
        let mut map_to = (code & 0xffff) as i16; // drop mode bits and sign-extend.
        let mode = ((code >> 8) as i16) >> 15; // sign-extend.

        // If the mask index refered by the action table exists also in the
        // token_masks index, then set mask_new to 0xffff and 0 otherwise.
        let mask_bit = code.to_be_bytes()[1] & 0x7f;
        let mask_idx = code.to_be_bytes()[0] as usize;
        let mask = e2_slice[mask_idx];
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
        map_to += mode & e2;

        // This works because token_masks are disjoint patterns. Ensuring
        // that only one will match.
        result += map_to & mask_val;
    }
    if result == 0 {
        result = default;
    }
    */
    result
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
        debug_assert!(t < TABLES.terminals_count);
        debug_assert!(state < TABLES.state_count);
        Action(dual_map(TABLES.action_token, state, t, ERROR))
    }

    fn goto_at_state(&self, nt : NonterminalId, state: usize) -> usize {
        // TABLES.goto_table[state * TABLES.goto_width + nt as usize] as usize
        let nt = nt as usize;
        debug_assert!(nt < TABLES.nonterminals_count);
        debug_assert!(state < TABLES.state_count);
        dual_map(TABLES.goto_nt, state, nt, 0) as usize
    }

    fn reduce_all(&mut self, t: TerminalId) -> Result<'alloc, Action> {
        let tables = TABLES;
        let mut action = self.action(t);
        while action.is_reduce() {
            let prod_index = action.reduce_prod_index();
            let nt = reduce(&self.handler, prod_index, &mut self.node_stack)?;
            debug_assert!(self.state_stack.len() >= self.node_stack.len());
            self.state_stack.truncate(self.node_stack.len());
            let prev_state = *self.state_stack.last().unwrap();
            let state_after = self.goto_at_state(nt, prev_state);
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
            debug_assert!(self.state_stack.len() >= self.node_stack.len());
            sp = sp - num_pops;
            let prev_state = self.state_stack[sp];
            state = self.goto_at_state(nt, prev_state);
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
