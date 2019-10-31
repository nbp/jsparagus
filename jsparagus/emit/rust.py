"""Emit code and parser tables in Rust."""

import re
import unicodedata
import math
from collections import Counter

from ..runtime import (ACCEPT, ERROR, ErrorToken)
from ..ordered import OrderedSet

from ..grammar import (CallMethod, Some, is_concrete_element, Nt, Optional)

from .. import types


TERMINAL_NAMES = {
    "=>": "Arrow",
}

# List of output method names that are fallible and must therefore be called
# with a trailing `?`. A bad hack which we need to fix by having more type
# information about output methods.
FALLIBLE_METHOD_NAMES = {
    'assignment_expression',
    'async_arrow_parameters',
    'compound_assignment_expression',
    'expression_to_parameter_list',
    'for_assignment_target',
    'for_await_of_statement',
    'post_decrement_expr',
    'post_increment_expr',
    'pre_decrement_expr',
    'pre_increment_expr',
}

class ActionKind:
    """Store the compressed action kinds to be stored instead of the transition
    list."""
    def __init__(self, kind, mask, state):
        self.kind = kind
        self.mask = mask
        self.state = state

class RustParserWriter:
    def __init__(self, out, parser_states):
        self.out = out
        self.grammar = parser_states.grammar
        self.prods = parser_states.prods
        self.states = parser_states.states
        self.init_state_map = parser_states.init_state_map
        self.terminals = list(OrderedSet(
            t for state in self.states for t in state.action_row))
        self.nonterminals = list(OrderedSet(
            nt for state in self.states for nt in state.ctn_row))

    def compress_actions(self):
        """Actions are a mapping of (state, terminal) to another state. However,
        it is highly redundant and can be compressed. Compression is useful to
        reduce the number of data cache misses while looking up the state
        transition.

        Each action has multiple forms of redundancy. The redundancy appears
        either as the same state transition (repeated state numbers) or as a
        sequence of incrementing state numbers.

        Some of the repeated patterns also appear in multiple times in
        different states. For example, the terminal `{` is frequently used as a
        terminal and appears in multiple state transitions. Thus a pattern
        matching all terminals expected by this state transition can be shared
        for each state.

        This function detects known forms of redundancy and attempts at
        encoding these efficiently, as well as generating all the data
        necessary for decoding it.

        """
        state_index = []
        transition_list = []
        mask_dict = {}
        mask_list = []
        for i, state in enumerate(self.states):
            state_index.append(len(transition_list))
            transition_table_ref = [state.action_row.get(t, -math.inf) for t in self.terminals]
            filtered = [t == -math.inf for t in transition_table_ref]

            stash_transition_list = []
            while filtered.count(False) > 0:
                transition_table = [ (-math.inf if f else s) for s, f in zip(transition_table_ref, filtered) ]
                # Identify sequences of transitions by substracting the index,
                # by substracting the index, we get the offset at the origin.
                # If multiple state transition have the same offset at the
                # origin, then we can create a sequence mask.
                sequence_table = [ s - i for i, s in enumerate(transition_table) ]
                state_counts = Counter(transition_table)
                seq_counts = Counter(sequence_table)
                del state_counts[-math.inf]
                del seq_counts[-math.inf]
                state, state_count = state_counts.most_common(1)[0]
                seq, seq_count = seq_counts.most_common(1)[0]
                if state_count >= seq_count:
                    mask = [ state == s for s in transition_table ]
                else:
                    mask = [ seq == s - i for i, s in enumerate(transition_table) ]

                # Reverse the bit string as low bits are written last.
                mask_str = ''.join(reversed([ str(int(b)) for b in mask ]))
                if mask_str in mask_dict:
                    mask_dict[mask_str]['count'] += 1
                    mask_idx = mask_dict[mask_str]['reg_idx']
                else:
                    mask_idx = len(mask_dict)
                    mask_dict[mask_str] = { 'reg_idx' : mask_idx, 'new_idx': 0, 'count': 1 }
                    mask_list.append(mask_str)

                if state_count >= seq_count:
                    stash_transition_list.append(ActionKind('Repeat', mask_idx, state))
                else:
                    stash_transition_list.append(ActionKind('Sequence', mask_idx, seq))

                filtered = [ m or f for m, f in zip(mask, filtered) ]

            stash_transition_list.sort(key = lambda x: x.mask)
            transition_list = transition_list + stash_transition_list

        # Sort masks based on the number of time they are reused, such that the
        # first mask to match are encountered first.

        # mask_list.sort(key = lambda m: -mask_dict[m]['count'])
        # mask_map = [ 0 for _ in mask_list ]
        # for idx, mask in enumerate(mask_list):
        #     mask_dict[mask]['new_idx'] = idx
        #     mask_map[mask_dict[mask]['reg_idx']] = idx
        # for act in transition_list:
        #     act.mask = mask_map[act.mask]

        # Invert the mask to change from `table[mask] = token-list`, to be a
        # `table[token] = mask-list`.
        token_masks = [''.join(reversed([ m[-1 - t] for m in mask_list ])) for t, _ in enumerate(self.terminals)]

        # The state_index contains absolute indexes to the transition_table.
        # However, to make the encoding of the state_index vector smaller, by
        # using relative indexes.
        state_index.append(len(transition_list))
        self.transition_list = transition_list
        self.state_index = state_index
        self.token_masks = token_masks
        self.mask_list = mask_list

    def emit(self):
        self.header()
        self.terminal_id()
        self.token()
        self.actions()
        self.error_codes()
        self.check_camel_case()
        self.nonterminal_id()
        self.goto()
        self.reduce()
        self.reduce_simulator()
        self.entry()

    def write(self, indentation, string, *format_args):
        if len(format_args) == 0:
            formatted = string
        else:
            formatted = string.format(*format_args)
        self.out.write("    " * indentation + formatted + "\n")

    def header(self):
        self.write(0, "// WARNING: This file is autogenerated.")
        self.write(0, "")
        self.write(0, "use ast::{arena::{Box, Vec}, types::*};")
        self.write(0, "use crate::ast_builder::AstBuilder;")
        self.write(0, "use crate::stack_value_generated::{StackValue, TryIntoStack};")
        self.write(0, "use crate::error::Result;")
        self.write(0, "")

    def terminal_name(self, value):
        if value is None:
            return "End"
        elif value is ErrorToken:
            return "ErrorToken"
        elif value in TERMINAL_NAMES:
            return TERMINAL_NAMES[value]
        elif value.isalpha():
            if value.islower():
                return value.capitalize()
            else:
                return value
        else:
            raw_name = " ".join((unicodedata.name(c) for c in value))
            snake_case = raw_name.replace("-", " ").replace(" ", "_").lower()
            camel_case = self.to_camel_case(snake_case)
            return camel_case

    def terminal_name_camel(self, value):
        return self.to_camel_case(self.terminal_name(value))

    def terminal_id(self):
        self.write(0, "#[derive(Copy, Clone, Debug, PartialEq)]")
        self.write(0, "pub enum TerminalId {")
        for i, t in enumerate(self.terminals):
            name = self.terminal_name(t)
            self.write(1, "{} = {}, // {}", name, i, repr(t))
        self.write(0, "}")
        self.write(0, "")

    def token(self):
        self.write(0, "#[derive(Clone, Debug, PartialEq)]")
        self.write(0, "pub struct Token<'a> {")
        self.write(1, "pub terminal_id: TerminalId,")
        self.write(1, "pub offset: usize,")
        self.write(1, "pub saw_newline: bool,")
        self.write(1, "pub value: Option<&'a str>,")
        self.write(0, "}")
        self.write(0, "")

        self.write(0, "impl Token<'_> {")
        self.write(1, "pub fn basic_token(terminal_id: TerminalId, offset: usize) -> Self {")
        self.write(2, "Self {")
        self.write(3, "terminal_id,")
        self.write(3, "offset,")
        self.write(3, "saw_newline: false,")
        self.write(3, "value: None,")
        self.write(2, "}")
        self.write(1, "}")
        self.write(0, "")

        self.write(1, "pub fn into_static(self) -> Token<'static> {")
        self.write(2, "Token {")
        self.write(3, "terminal_id: self.terminal_id,")
        self.write(3, "offset: self.offset,")
        self.write(3, "saw_newline: self.saw_newline,")
        self.write(3, "value: None,")  # drop the value, which has limited lifetime
        self.write(2, "}")
        self.write(1, "}")
        self.write(0, "}")
        self.write(0, "")

    def actions(self):
        self.compress_actions()
        # Split the token masks by u64, such that we can load parts of the
        # mask-bits.
        bits = 32
        self.masks_bits = bits
        token_masks = []
        for masks in self.token_masks:
            while masks != '':
                token_masks.append(masks[-bits:])
                masks = masks[:-bits]
        masks_per_token = int(len(token_masks) / len(self.token_masks))

        self.write(0, "#[rustfmt::skip]")
        self.write(0, "const MASKS_PER_TOKEN: usize = {};", masks_per_token)
        self.write(0, "")
        self.write(0, "#[rustfmt::skip]")
        self.write(0, "const TOKEN_MASKS: [u{}; {}] = [", bits, len(token_masks))
        for i, masks in enumerate(token_masks):
            if i % masks_per_token == 0:
                if i > 0:
                    self.write(0, "")
                tok = int(i / masks_per_token)
                self.write(1, "// {}. token: {}", tok, self.terminals[tok])
                self.write(1, "// masks: {}", ', '.join([str(i) for i, b in enumerate(reversed(self.token_masks[tok])) if b == '1']))

            self.write(1, "0b{},", masks)
        self.write(0, "];")
        self.write(0, "")
        self.write(0, "#[rustfmt::skip]")
        self.write(0, "const ACTIONS: [u32; {}] = [", len(self.transition_list))
        state_id = 0
        for i, transition in enumerate(self.transition_list):
            # IF we are reaching the entry for a given state, add a comment.
            if self.state_index[state_id] == i:
                self.write(1, "// {}. {}", state_id, self.states[state_id].traceback() or "<empty>")
                state_id += 1
            # self.write(1, "({} << 31) | ({} << 16) | ({}i16 as u16 as u32),",
            #            '1' if transition.kind == "Sequence" else '0',
            #            transition.mask,
            #            transition.state if transition.state != ACCEPT else '-0x7fff')
            self.write(1, "({} << 24) | ({} << 23) | ({} << 16) | ({}i16 as u16 as u32),",
                       int(transition.mask / bits),
                       '1' if transition.kind == "Sequence" else '0',
                       transition.mask % bits,
                       transition.state if transition.state != ACCEPT else '-0x7fff')
        self.write(0, "];")
        self.write(0, "")
        self.write(0, "#[rustfmt::skip]")
        self.write(0, "const ACTION_IDX: [u16; {}] = [", len(self.state_index))
        for i, offset in enumerate(self.state_index):
            self.write(1, "{},", offset)
        self.write(0, "];")
        self.write(0, "")

    def error_codes(self):
        self.write(0, "#[derive(Clone, Debug, PartialEq)]")
        self.write(0, "pub enum ErrorCode {")
        for error_code in OrderedSet(s.error_code for s in self.states):
            if error_code is not None:
                self.write(1, "{},", self.to_camel_case(error_code))
        self.write(0, "}")
        self.write(0, "")

        self.write(0, "const STATE_TO_ERROR_CODE: [Option<ErrorCode>; {}] = [",
                   len(self.states))
        for i, state in enumerate(self.states):
            self.write(1, "// {}. {}", i, state.traceback() or "<empty>")
            if state.error_code is None:
                self.write(1, "None,")
            else:
                self.write(1, "Some(ErrorCode::{}),",
                           self.to_camel_case(state.error_code))
        self.write(0, "];")
        self.write(0, "")

    def nonterminal_to_snake(self, ident):
        if isinstance(ident, Nt):
            base_name = self.to_snek_case(ident.name)
            args = ''.join((("_" + self.to_snek_case(name))
                            for name, value in ident.args if value))
            return base_name + args
        else:
            assert isinstance(ident, str)
            return self.to_snek_case(ident)

    def nonterminal_to_camel(self, nt):
        return self.to_camel_case(self.nonterminal_to_snake(nt))

    def to_camel_case(self, ident):
        if '_' in ident:
            return ''.join(word.capitalize() for word in ident.split('_'))
        elif ident.islower():
            return ident.capitalize()
        else:
            return ident

    def check_camel_case(self):
        seen = {}
        for nt in self.nonterminals:
            cc = self.nonterminal_to_camel(nt)
            if cc in seen:
                raise ValueError("{} and {} have the same camel-case spelling ({})".format(
                    seen[cc], nt, cc))
            seen[cc] = nt

    def to_snek_case(self, ident):
        # https://stackoverflow.com/questions/1175208
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', ident)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def method_name_to_rust(self, name):
        """Convert jsparagus's internal method name to idiomatic Rust."""
        nt_name, space, number = name.partition(' ')
        name = self.nonterminal_to_snake(nt_name)
        if space:
            name += "_p" + str(number)
        return name

    def get_associated_type_names(self):
        names = OrderedSet()

        def visit_type(ty):
            for arg in ty.args:
                visit_type(arg)
            if len(ty.args) == 0:
                names.add(ty.name)

        for ty in self.grammar.nt_types:
            visit_type(ty)
        for method in self.grammar.methods.values():
            visit_type(method.return_type)
        return names

    def type_to_rust(self, ty, namespace, boxed=False):
        """
        Convert a jsparagus type (see types.py) to Rust.

        Pass boxed=True if the type needs to be boxed.
        """
        if ty == types.UnitType:
            assert not boxed
            rty = '()'
        elif ty == types.TokenType:
            rty = "Token<'alloc>"
        elif ty.name == 'Option' and len(ty.args) == 1:
            # We auto-translate `Box<Option<T>>` to `Option<Box<T>>` since
            # that's basically the same thing but more efficient.
            [arg] = ty.args
            return 'Option<{}>'.format(self.type_to_rust(arg, namespace, boxed))
        elif ty.name == 'Vec' and len(ty.args) == 1:
            [arg] = ty.args
            rty = "Vec<'alloc, {}>".format(self.type_to_rust(arg, namespace, boxed=False))
        else:
            if namespace == "":
                rty = ty.name
            else:
                rty = namespace + '::' + ty.name
            if ty.args:
                rty += '<{}>'.format(', '.join(self.type_to_rust(arg, namespace, boxed)
                                               for arg in ty.args))
        if boxed:
            return "Box<'alloc, {}>".format(rty)
        else:
            return rty

    def handler_trait(self):
        # NOTE: unused, code kept if we need it later
        self.write(0, "pub trait Handler {")

        for name in self.get_associated_type_names():
            self.write(1, "type {};", name)

        for tag, method in self.grammar.methods.items():
            method_name = self.method_name_to_rust(tag)
            arg_types = [
                self.type_to_rust(ty, "Self")
                for ty in method.argument_types
                if ty != types.UnitType
            ]
            if method.return_type == types.UnitType:
                return_type_tag = ''
            else:
                return_type_tag = ' -> ' + \
                    self.type_to_rust(method.return_type, "Self")

            args = ", ".join(("a{}: {}".format(i, t)
                              for i, t in enumerate(arg_types)))
            self.write(1, "fn {}(&self, {}){};",
                       method_name, args, return_type_tag)
        self.write(0, "}")
        self.write(0, "")

    def nonterminal_id(self):
        self.write(0, "#[derive(Clone, Copy, Debug, PartialEq)]")
        self.write(0, "pub enum NonterminalId {")
        for i, nt in enumerate(self.nonterminals):
            self.write(1, "{} = {},", self.nonterminal_to_camel(nt), i)
        self.write(0, "}")
        self.write(0, "")

    def goto(self):
        self.write(0, "#[rustfmt::skip]")
        self.write(0, "const GOTO: [u16; {}] = [",
                   len(self.states) * len(self.nonterminals))
        for state in self.states:
            row = state.ctn_row
            self.write(1, "{}", ' '.join("{},".format(row.get(nt, 0))
                                         for nt in self.nonterminals))
        self.write(0, "];")
        self.write(0, "")

    def element_type(self, e):
        # Mostly duplicated from types.py. :(
        g = self.grammar
        if isinstance(e, str):
            if e in g.variable_terminals:
                return types.TokenType
            else:
                # constant terminal
                return types.UnitType
        elif isinstance(e, Optional):
            return types.Type('Option', [self.element_type(e.inner)])
        elif isinstance(e, Nt):
            # Cope with the awkward fact that g.nonterminals keys may be either
            # strings or Nt objects.
            nt_key = e if e in g.nonterminals else e.name
            assert g.nonterminals[nt_key].type is not None
            return g.nonterminals[nt_key].type
        else:
            assert False, "unexpected element type: {!r}".format(e)

    def reduce(self):
        # Note use of std::vec::Vec below: we have imported `arena::Vec` in this module,
        # since every other data structure mentioned in this file lives in the arena.
        self.write(0, "pub fn reduce<'alloc>(")
        self.write(1, "handler: &AstBuilder<'alloc>,")
        self.write(1, "prod: usize,")
        self.write(1, "stack: &mut std::vec::Vec<StackValue<'alloc>>,")
        self.write(0, ") -> Result<'alloc, NonterminalId> {")
        self.write(1, "match prod {")
        for i, prod in enumerate(self.prods):
            # If prod.nt is not in nonterminals, that means it's a goal
            # nonterminal, only accepted, never reduced.
            if prod.nt in self.nonterminals:
                self.write(2, "{} => {{", i)
                self.write(3, "// {}",
                           self.grammar.production_to_str(prod.nt, prod.rhs, prod.reducer))

                # At run time, the top of the stack will be one value per
                # concrete symbol in the RHS of the production we're reducing.
                # We are about to emit code to pop these values from the stack,
                # one at a time. They come off the stack in reverse order.
                elements = [e for e in prod.rhs if is_concrete_element(e)]

                # We can emit three different kinds of code here:
                #
                # 1.  Full compilation. Pop each value from the stack; if it's
                #     used, downcast it to its actual type and store it in a
                #     local variable (otherwise just drop it). Then, evaulate
                #     the reduce-expression. Push the result back onto the
                #     stack.
                #
                # 2.  `is_discarding_reduction`: A reduce expression that is
                #     just an integer is retaining one stack value and dropping
                #     the rest. We skip the downcast in this case.
                #
                # 3.  `is_trivial_reduction`: A production has only one
                #     concrete symbol in it, and the reducer is just `0`.
                #     We don't have to do anything at all here.
                is_trivial_reduction = len(elements) == 1 and prod.reducer == 0
                is_discarding_reduction = isinstance(prod.reducer, int)

                # While compiling, figure out which elements are used.
                variable_used = [False] * len(elements)

                def compile_reduce_expr(expr):
                    """Compile a reduce expression to Rust"""
                    if isinstance(expr, CallMethod):
                        method_type = self.grammar.methods[expr.method]
                        method_name = self.method_name_to_rust(expr.method)
                        assert len(method_type.argument_types) == len(expr.args)
                        args = ', '.join(
                            compile_reduce_expr(arg)
                            for ty, arg in zip(method_type.argument_types,
                                               expr.args)
                            if ty != types.UnitType)
                        call = "handler.{}({})".format(method_name, args)

                        # Extremely bad hack. In Rust, since type inference is
                        # currently so poor, we don't have enough information
                        # to know if this method can fail or not, and Rust
                        # requires us to know that.
                        if method_name in FALLIBLE_METHOD_NAMES:
                            call += "?"
                        return call
                    elif isinstance(expr, Some):
                        return "Some({})".format(compile_reduce_expr(expr.inner))
                    elif expr is None:
                        return "None"
                    else:
                        # can't be 'accept' because we filter out InitNt productions
                        assert isinstance(expr, int)
                        variable_used[expr] = True
                        return "x{}".format(expr)

                compiled_expr = compile_reduce_expr(prod.reducer)

                if not is_trivial_reduction:
                    for index, e in reversed(list(enumerate(elements))):
                        if variable_used[index]:
                            ty = self.element_type(e)
                            rust_ty = self.type_to_rust(ty, "", boxed=True)
                            if is_discarding_reduction:
                                self.write(3, "let x{} = stack.pop().unwrap();", index)
                            else:
                                self.write(3, "let x{}: {} = stack.pop().unwrap().to_ast();", index, rust_ty)
                        else:
                            self.write(3, "stack.pop();", index)

                    if is_discarding_reduction:
                        self.write(3, "stack.push({});", compiled_expr)
                    else:
                        self.write(3, "stack.push(TryIntoStack::try_into_stack({})?);", compiled_expr)

                self.write(3, "Ok(NonterminalId::{})",
                           self.nonterminal_to_camel(prod.nt))
                self.write(2, "}")
        self.write(2, '_ => panic!("no such production: {}", prod),')
        self.write(1, "}")
        self.write(0, "}")
        self.write(0, "")

    def reduce_simulator(self):
        prods = [prod for prod in self.prods if prod.nt in self.nonterminals]
        self.write(0, "const REDUCE_SIMULATOR: [(usize, NonterminalId); {}] = [", len(prods))
        for prod in prods:
            elements = [e for e in prod.rhs if is_concrete_element(e)]
            self.write(1, "({}, NonterminalId::{}),", len(elements), self.nonterminal_to_camel(prod.nt))
        self.write(0, "];")
        self.write(0, "")

    def entry(self):
        self.write(0, "#[derive(Clone, Copy)]")
        self.write(0, "pub struct ParserTables<'a> {")
        self.write(1, "pub state_count: usize,")
        self.write(1, "pub masks_per_token: usize,")
        self.write(1, "pub token_masks: &'a [u{}],", self.masks_bits)
        self.write(1, "pub action_table: &'a [u32],")
        self.write(1, "pub action_idx: &'a [u16],")
        self.write(1, "pub action_width: usize,")
        self.write(1, "pub error_codes: &'a [Option<ErrorCode>],")
        self.write(1, "pub reduce_simulator: &'a [(usize, NonterminalId)],")
        self.write(1, "pub goto_table: &'a [u16],")
        self.write(1, "pub goto_width: usize,")
        self.write(0, "}")
        self.write(0, "")

        self.write(0, "impl<'a> ParserTables<'a> {")
        self.write(1, "pub fn check(&self) {")
        self.write(2, "assert_eq!(self.goto_table.len(), (self.state_count * self.goto_width) as usize);")
        self.write(1, "}")
        self.write(0, "}")
        self.write(0, "")

        self.write(0, "pub const TABLES: ParserTables<'static> = ParserTables {")
        self.write(1, "state_count: {},", len(self.states))
        self.write(1, "masks_per_token: MASKS_PER_TOKEN,")
        self.write(1, "token_masks: &TOKEN_MASKS,")
        self.write(1, "action_table: &ACTIONS,")
        self.write(1, "action_idx: &ACTION_IDX,")
        self.write(1, "action_width: {},", len(self.terminals))
        self.write(1, "error_codes: &STATE_TO_ERROR_CODE,")
        self.write(1, "reduce_simulator: &REDUCE_SIMULATOR,")
        self.write(1, "goto_table: &GOTO,")
        self.write(1, "goto_width: {},".format(len(self.nonterminals)))
        self.write(0, "};")
        self.write(0, "")

        for init_nt, index in self.init_state_map.items():
            assert init_nt.args == ()
            self.write(0, "pub static START_STATE_{}: usize = {};",
                       self.to_snek_case(init_nt.name).upper(), index)
            self.write(0, "")


def write_rust_parser(out, parser_states):
    RustParserWriter(out, parser_states).emit()
