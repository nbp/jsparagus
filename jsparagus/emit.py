"""Emit code for parser tables in either Python or Rust. """

import re
import unicodedata

from .runtime import ERROR
from .ordered import OrderedSet
from .grammar import InitNt, CallMethod, Some, is_lookahead_rule, is_apply, is_concrete_element, Optional
from . import types


def write_python_parser(out, grammar, states, prods, init_state_map):
    out.write("from jsparagus import runtime\n")
    if any(is_apply(key) for key in grammar.nonterminals):
        out.write("from jsparagus.runtime import Apply\n")
    out.write("\n")

    out.write("actions = [\n")
    for i, state in enumerate(states):
        out.write("    # {}. {}\n".format(i, state.traceback() or "<empty>"))
        # for item in state._lr_items:
        #     out.write("    #       {}\n".format(grammar.lr_item_to_str(prods, item)))
        out.write("    " + repr(state.action_row) + ",\n")
        out.write("\n")
    out.write("]\n\n")
    out.write("ctns = [\n")
    for state in states:
        out.write("    " + repr(state.ctn_row) + ",\n")
    out.write("]\n\n")

    def action(a):
        """Compile a reduce expression to Python"""
        if isinstance(a, CallMethod):
            method_name = a.method.replace(" ", "_P")
            return "builder.{}({})".format(method_name, ', '.join(map(action, a.args)))
        elif isinstance(a, Some):
            return action(a.inner)
        elif a is None:
            return "None"
        else:
            # can't be 'accept' because we filter out InitNt productions
            assert isinstance(a, int)
            return "x{}".format(a)

    out.write("reductions = [\n")
    for prod in prods:
        if isinstance(prod.nt, InitNt):
            continue
        nparams = sum(1 for e in prod.rhs if is_concrete_element(e))
        names = ["x" + str(i) for i in range(nparams)]
        fn = ("lambda builder, "
              + ", ".join(names)
              + ": " + action(prod.action))
        out.write("    ({!r}, {!r}, {}),\n".format(prod.nt, len(names), fn))
    out.write("]\n\n\n")  # two blank lines before class.

    out.write("class DefaultBuilder:\n")
    for tag, method_type in grammar.methods.items():
        method_name = tag.replace(' ', '_P')
        args = ", ".join("x{}".format(i)
                         for i in range(len(method_type.argument_types)))
        out.write("    def {}(self, {}): return ({!r}, {})\n"
                  .format(method_name, args, tag, args))
    out.write("\n\n")

    for init_nt, index in init_state_map.items():
        out.write("parse_{} = runtime.make_parse_fn(actions, ctns, reductions, {}, DefaultBuilder)\n"
                  .format(init_nt, index))


TERMINAL_NAMES = {
    "=>": "Arrow",
}


class RustParserWriter:
    def __init__(self, out, grammar, states, prods, init_state_map):
        self.out = out
        self.grammar = grammar
        self.states = states
        self.prods = prods
        self.init_state_map = init_state_map
        self.terminals = list(OrderedSet(
            t for state in self.states for t in state.action_row))
        self.nonterminals = list(OrderedSet(
            nt for state in self.states for nt in state.ctn_row))

        self.prod_optional_element_indexes = {
            (prod.nt, prod.index): set(
                i
                for p in self.prods
                if p.nt == prod.nt and p.index == prod.index
                for i in p.removals
            )
            for prod in self.prods
            if prod.nt in self.nonterminals and not prod.removals
        }

        # Reverse-engineered original productions for everything
        self.originals = {
            (prod.nt, prod.index): [
                (Optional(e)
                 if i in self.prod_optional_element_indexes[prod.nt, prod.index] else e)
                for i, e in enumerate(prod.rhs)
            ]
            for prod in self.prods
            if prod.nt in self.nonterminals and not prod.removals
        }

    def emit(self):
        self.header()
        self.terminal_id()
        self.token()
        self.actions()
        self.check_camel_case()
        self.check_nt_node_variant()
        self.handler_trait()
        # self.nt_node()
        # self.nt_node_impl()
        self.nonterminal_id()
        self.goto()
        self.reduce()
        self.entry()

    def header(self):
        self.out.write("// THIS FILE IS AUTOGENERATED -- HAHAHAHA\n\n")

        self.out.write(
            "use super::parser_runtime::{self, ParserTables, TokenStream};\n\n")

        self.out.write("const ERROR: i64 = {};\n\n".format(hex(ERROR)))

    def terminal_name(self, value):
        if value is None:
            return "End"
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
        self.out.write("#[derive(Copy, Clone, Debug, PartialEq)]\n")
        self.out.write("pub enum TerminalId {\n")
        for i, t in enumerate(self.terminals):
            name = self.terminal_name(t)
            self.out.write("    {} = {}, // {}\n".format(name, i, repr(t)))
        self.out.write("}\n\n")

    def token(self):
        self.out.write("#[derive(Clone, Debug, PartialEq)]\n")
        self.out.write("pub enum Token {\n")
        for i, t in enumerate(self.terminals):
            name = self.terminal_name(t)
            value = "(String)" if t in self.grammar.variable_terminals else ""
            self.out.write("    {}{}, // {}\n".format(name, value, repr(t)))
        self.out.write("}\n\n")

        self.out.write("impl Token {\n")
        self.out.write("    pub fn get_id(&self) -> TerminalId {\n")
        self.out.write("        // This switch should be optimized away.\n")
        self.out.write("        match self {\n")
        for i, t in enumerate(self.terminals):
            name = self.terminal_name(t)
            value = "(_)" if t in self.grammar.variable_terminals else ""
            self.out.write(
                "            Token::{}{} => TerminalId::{},\n".format(name, value, name))
        self.out.write("        }\n")
        self.out.write("    }\n")
        self.out.write("}\n\n")

    def actions(self):
        self.out.write("static ACTIONS: [i64; {}] = [\n".format(
            len(self.states) * len(self.terminals)))
        for i, state in enumerate(self.states):
            self.out.write("    // {}. {}\n".format(i,
                                                    state.traceback() or "<empty>"))
            self.out.write("    {}\n".format(' '.join("{},".format(state.action_row.get(t, "ERROR"))
                                                      for t in self.terminals)))
            if i < len(self.states) - 1:
                self.out.write("\n")
        self.out.write("];\n\n")

    def nonterminal_to_snake(self, ident):
        if is_apply(ident):
            base_name = self.to_snek_case(ident.nt)
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

    def nt_node_variant(self, prod):
        name = self.nonterminal_to_camel(prod.nt)
        if len(self.grammar.nonterminals[prod.nt]) > 1:
            name += "P" + str(prod.index)
        return name

    def check_nt_node_variant(self):
        seen = {}
        for prod in self.prods:
            if prod.nt in self.nonterminals and not prod.removals:
                name = self.nt_node_variant(prod)
                if name in seen:
                    raise ValueError("Productions {} and {} have the same spelling ({})".format(
                        self.grammar.production_to_str(
                            seen[name].nt, seen[name].rhs),
                        self.grammar.production_to_str(prod.nt, prod.rhs),
                        name))
                seen[name] = prod

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
            if isinstance(ty, types.NtType):
                names.add(ty.name)
            elif isinstance(ty, types.OptionType):
                visit_type(ty.t)

        for ty in self.grammar.nt_types:
            visit_type(ty)
        for method in self.grammar.methods.values():
            visit_type(method.return_type)
        return names

    def type_to_rust(self, ty, handler):
        """Convert a jsparagus type (see types.py) to Rust."""
        if ty is types.UnitType:
            return '()'
        elif ty == 'str':
            return 'String'
        elif ty == 'bool':
            return 'bool'
        elif isinstance(ty, types.NtType):
            return handler + '::' + ty.name
        elif isinstance(ty, types.OptionType):
            return 'Option<{}>'.format(self.type_to_rust(ty.t, handler))
        else:
            raise TypeError("unexpected type: {!r}".format(ty))

    def handler_trait(self):
        self.out.write("pub trait Handler {\n")

        for name in self.get_associated_type_names():
            self.out.write("    type {};\n".format(name))

        for tag, method in self.grammar.methods.items():
            method_name = self.method_name_to_rust(tag)
            arg_types = [
                self.type_to_rust(ty, "Self")
                for ty in method.argument_types
                if ty != types.UnitType
            ]
            if method.return_type is types.UnitType:
                return_type_tag = ''
            else:
                return_type_tag = ' -> ' + \
                    self.type_to_rust(method.return_type, "Self")

            args = ", ".join(("a{}: {}".format(i, t)
                              for i, t in enumerate(arg_types)))
            self.out.write(
                "    fn {}(&mut self, {}){};\n"
                .format(method_name, args, return_type_tag))
        self.out.write("}\n\n")

    def nt_node(self):
        self.out.write("#[derive(Debug)]\n")
        self.out.write("pub enum NtNode {\n")
        for prod in self.prods:
            if prod.nt in self.nonterminals and not prod.removals:
                types = []
                for i, e in enumerate(prod.rhs):
                    ty = self.rust_type_of_element(prod, i, e, "NtNode")
                    if ty is not None and ty != '()':
                        types.append(ty)

                self.out.write(
                    "    // {}\n".format(self.grammar.production_to_str(prod.nt, prod.rhs)))
                name = self.nt_node_variant(prod)
                self.out.write("    {}({}),\n".format(name, ", ".join(types)))
        self.out.write("}\n\n")

    def nt_node_impl(self):
        self.out.write("pub struct DefaultHandler {}\n\n")
        self.out.write("impl Handler for DefaultHandler {\n")
        self.out.write("    // TODO\n")
        for method in self.grammar.methods:
            pass
            # if prod.nt in self.nonterminals and not prod.removals:
            #     types = []
            #     for i, e in enumerate(prod.rhs):
            #         ty = self.rust_type_of_element(prod, i, e, "NtNode")
            #         if ty is not None and ty != '()':
            #             types.append(ty)
            #
            #     method_name = ?
            #     nt_node_name = self.nt_node_variant(prod)
            #     params = ", ".join(("a{}: {}".format(i, t)
            #                         for i, t in enumerate(types)))
            #     args = ", ".join("a{}".format(i)
            #                      for i in range(0, len(types)))
            #     self.out.write(
            #         "    fn {}(&mut self, {}) -> NtNode {{\n".format(method_name, params))
            #     self.out.write("        NtNode::{}({})\n" .format(
            #         nt_node_name, args))
            #     self.out.write("    }\n")
        self.out.write("}\n\n")

    def nonterminal_id(self):
        self.out.write("#[derive(Clone, Copy, Debug, PartialEq)]\n")
        self.out.write("pub enum NonterminalId {\n")
        for i, nt in enumerate(self.nonterminals):
            self.out.write("    {} = {},\n".format(
                self.nonterminal_to_camel(nt), i))
        self.out.write("}\n\n")

    def goto(self):
        self.out.write("static GOTO: [usize; {}] = [\n".format(
            len(self.states) * len(self.nonterminals)))
        for state in self.states:
            row = state.ctn_row
            self.out.write("    {}\n".format(' '.join("{},".format(row.get(nt, 0))
                                                      for nt in self.nonterminals)))
        self.out.write("];\n\n")

    def reduce(self):
        self.out.write(
            "fn reduce<H: Handler>(handler: &mut H, prod: usize, stack: &mut Vec<*mut ()>) -> NonterminalId {\n")
        self.out.write("    match prod {\n")
        for i, prod in enumerate(self.prods):
            # If prod.nt is not in nonterminals, that means it's a goal
            # nonterminal, only accepted, never reduced.
            if prod.nt in self.nonterminals:
                self.out.write("        {} => {{\n".format(i))
                self.out.write(
                    "            // {}\n".format(self.grammar.production_to_str(prod.nt, prod.rhs)))

                for index in range(len(prod.rhs)-1, -1, -1):
                    self.out.write(
                        "            let x{} = stack.pop().unwrap();\n".format(index))

                def compile_reduce_expr(expr):
                    """Compile a reduce expression to Rust"""
                    if isinstance(expr, CallMethod):
                        method_type = self.grammar.methods[expr.method]
                        method_name = self.method_name_to_rust(expr.method)
                        args = ', '.join(
                            compile_reduce_expr(arg)
                            for index, arg in enumerate(expr.args)
                            if method_type.argument_types[index] is not types.UnitType)
                        call = "handler.{}({})".format(method_name, args)
                        return "{}".format(call)
                    elif isinstance(expr, Some):
                        return "Some({})".format(compile_reduce_expr(expr.inner))
                    elif expr is None:
                        return "None"
                    else:
                        # can't be 'accept' because we filter out InitNt productions
                        assert isinstance(expr, int)
                        return "unsafe {{ *Box::from_raw(x{} as *mut _) }}".format(expr)

                self.out.write("            stack.push(Box::into_raw(Box::new({})) as *mut ());\n"
                               .format(compile_reduce_expr(prod.action)))
                self.out.write("            NonterminalId::{}\n"
                               .format(self.nonterminal_to_camel(prod.nt)))
                self.out.write("        }\n")
        self.out.write(
            '        _ => panic!("no such production: {}", prod),\n')
        self.out.write("    }\n")
        self.out.write("}\n\n")

    def entry(self):
        self.out.write(
            "static TABLES: ParserTables<'static> = ParserTables {\n" +
            "    state_count: {},\n".format(len(self.states)) +
            "    action_table: &ACTIONS,\n" +
            "    action_width: {},\n".format(len(self.terminals)) +
            "    goto_table: &GOTO,\n" +
            "    goto_width: {},\n".format(len(self.nonterminals)) +
            "};\n\n"
        )

        for init_nt, index in self.init_state_map.items():
            result_type_jsparagus = self.grammar.nt_types[init_nt]
            result_type = self.type_to_rust(result_type_jsparagus, "H")
            self.out.write(
                "pub fn parse_{}<H: Handler, In: TokenStream<Token = Token>>(\n".format(init_nt))
            self.out.write("    handler: &mut H,\n")
            self.out.write("    tokens: In,\n")
            self.out.write(
                ") -> Result<{}, &'static str> {{\n".format(result_type))
            self.out.write(
                "    let result = parser_runtime::parse(handler, tokens, {}, &TABLES, reduce)?;\n".format(index))
            self.out.write("Ok(Box::from_raw(result as *mut _))")
            self.out.write("}\n\n")
