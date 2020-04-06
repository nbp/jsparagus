import collections
from .actions import Action

# An edge in a Parse table is a tuple of a source state and the term followed
# to exit this state. The destination is not saved here as it can easily be
# inferred by looking it up in the parse table.
#
# Note, the term might be `None` if no term is specified yet. This is useful
# when manipulating a list of edges and we know that we are taking transitions
# from a given state, but not yet with which term.
#
#   state: Index of the state from which this directed edge is coming from.
#
#   term: Edge transition value, this can be a terminal, non-terminal or an
#       action to be executed on an epsilon transition.
Edge = collections.namedtuple("Edge", "src term")

def edge_str(edge):
    assert isinstance(edge, Edge)
    return "{} -- {} -->".format(edge.src, str(edge.term))

class APS:
    # To fix inconsistencies of the grammar, we have to traverse the grammar
    # both forward by using the lookahead and backward by using the state
    # recovered from following unwind actions.
    #
    # To do so we define the notion of abstract parser state (APS), which is a
    # class which represents the known state of the parser, relative to its
    # starting point.
    #
    # An APS does not exclusively start at the parser entry point, but starts
    # from any state of the parse table by calling `APS.start`. Then we walk
    # the parse table forward, as-if we were shifting tokens or epsilon edges
    # in the parse table. The function `aps.shift_next(parse_table)` will
    # explore all possible futures reachable from the starting point.
    #
    # As the parse table is explored, new APS are produced by
    # `aps.shift_next(parse_table)`, which are containing the new state of the
    # parser and the history which has been seen by the APS since it started.
    #
    #   stack: This is the known stack at the location where we started
    #          investigating. As more history is discovered by resolving unwind
    #          actions, this stack would be filled with the predecessors which
    #          have been visited before reaching the starting state.
    #
    #   shift: This is the stack as manipulated by an LR parser. States are
    #          shifted to it, including actions, and popped from it when
    #          visiting a unwind action.
    #
    #   lookahead: This is the list of terminals and non-terminals encountered
    #          by shifting edges which are not replying tokens.
    #
    #   replay: This is the list of lookahead terminals and non-terminals which
    #          remains to be shifted. This list corresponds to terminals and
    #          non-terminals which were necessary for removing inconsistencies,
    #          but have to be replayed after shifting the reduced
    #          non-terminals.
    #
    #   reducing: This is a boolean flag which is used to distinguish whether
    #          the next term to be replayed is the result of a Reduce action or
    #          not. When reducing, epsilon transitions should be ignored. This
    #          flag is useful to implement Unwind and Reduce as 2 different
    #          actions.
    #
    #   history: This is the list of edges visited since the starting state.
    #
    slots = 'stack', 'shift', 'lookahead', 'replay', 'reducing', 'history'
    def __init__(self, state, st, sh, la, rp, hs, reducing=False):
        self.state = state
        self.stack = st
        self.shift = sh
        self.lookahead = la
        self.replay = rp
        self.reducing = reducing
        self.history = hs
        assert self.is_valid_self()

    def is_valid_self(self):
        "Returns whether this structure contains the right content."
        check = True
        # self.state == pt.states[self.history[-1].src][self.history[-1].term]
        check &= isinstance(self.state, int)
        check &= all(isinstance(st, Edge) for st in self.stack)
        check &= all(isinstance(sh, Edge) for sh in self.shift)
        check &= self.stack[0].src == self.shift[0].src
        check &= all(not isinstance(la, Action) for la in self.lookahead)
        check &= all(not isinstance(rp, Action) for rp in self.replay)
        check &= isinstance(self.reducing, bool)
        check &= all(isinstance(ac, Edge) for ac in self.history)
        return check

    @staticmethod
    def start(state):
        "Return an Abstract Parser State starting at a given state of a parse table"
        edge = Edge(state, None)
        return APS(state, [edge], [edge], [], [], [])

    def shift_next(self, pt):
        """Visit all the states of the parse table, as-if we were running a
        Generalized LR parser.

        However, instead parsing content, we use this algorithm to generate
        both the content which remains to be parsed as well as the context
        which might lead us to be in the state which from which we started.

        This algorithm takes an APS (Abstract Parser State), and consider all
        edges of the parse table, unless restricted by one of the previously
        encountered actions. These restrictions, such as replayed lookahead or
        the path which might be reduced are used for filtering out states which
        are not handled by this parse table.

        For each edge, this functions recursively calls it-self and calls the
        visit functions to know whether to stop or continue, and to capture the
        result.

        """
        st, sh, la, rp, hs = self.stack, self.shift, self.lookahead, self.replay, self.history
        state = pt.states[self.state]
        state_match_shift_end = self.state == self.shift[-1].src
        if self.replay == []:
            assert state_match_shift_end
            for term, to in state.shifted_edges():
                edge = Edge(self.state, term)
                new_sh = self.shift[:-1] + [edge]
                to = Edge(to, None)
                yield APS(to.src, st, new_sh + [to], la + [term], rp, hs + [edge])
        elif state_match_shift_end:
            term = self.replay[0]
            rp = self.replay[1:]
            if term in state:
                edge = Edge(self.state, term)
                new_sh = self.shift[:-1] + [edge]
                to = state[term]
                to = Edge(to, None)
                yield APS(to.src, st, new_sh + [to], la, rp, hs + [edge])

        if self.reducing:
            assert state_match_shift_end
            # When reducing, do not attempt to execute epsilon actions. As
            # reduce actions are split into Unwind and replay, we need to
            # distinguish whether the replayed term is coming from a reduce
            # action. Without this flag, we might loop on Optional rules. Which
            # would not match the expected behaviour.
            return

        term = None
        rp = self.replay
        for a, to in state.epsilon:
            edge = Edge(self.state, a)
            prev_sh = self.shift[:-1] + [edge]
            # TODO: Add support for Lookahead and flag manipulation rules, as
            # both of these would invalidate potential reduce paths.
            if a.update_stack() and a.update_stack_with() == (0, None, -1):
                assert len(self.replay) >= 1
                term = self.replay[0]
                rp = self.replay[1:]
                sh_state = self.shift[-1].src
                sh_edge = Edge(sh_state, term)
                sh_to = pt.states[sh_state][term]
                sh_to = Edge(sh_to, None)
                new_sh = self.shift[:-1] + [sh_edge, sh_to]
                new_hs = hs + [hs_edge]
                assert sh_to.src == a.replay_dest
                yield APS(to, st, new_sh, la, rp, hs + [edge])
            elif a.update_stack():
                reducing = not a.follow_edge()
                pop, nt, replay = a.update_stack_with()
                assert pop >= 0
                assert nt is not None
                assert replay >= 0
                for path, reduced_path in pt.reduce_path(prev_sh):
                    # reduce_paths contains the chains of state shifted,
                    # including epsilon transitions, in order to reduce the
                    # nonterminal. When reducing, the stack is resetted to
                    # head, and the nonterminal `term.nt` is pushed, to resume
                    # in the state `to`.

                    # print("Compare shifted path, with reduced path:\n\tshifted = {}\n\treduced = {}, \n\taction = {},\n\tnew_path = {}\n".format(
                    #     " ".join(edge_str(e) for e in prev_sh),
                    #     " ".join(edge_str(e) for e in path),
                    #     str(a),
                    #     " ".join(edge_str(e) for e in reduced_path),
                    # ))
                    if prev_sh[-len(path):] != path[-len(prev_sh):]:
                        # If the reduced production does not match the shifted
                        # state, then this reduction does not apply. This is
                        # the equivalent result as splitting the parse table
                        # based on the predecessor.
                        continue

                    # The stack corresponds to the stack present at the
                    # starting point. The shift list correspond to the actual
                    # parser stack as we iterate through the state machine.
                    # Each time we consume all the shift list, this implies
                    # that we had extra stack elements which were not present
                    # initially, and therefore we are learning about the
                    # context.
                    new_st = path[:max(len(path) - len(prev_sh), 0)] + st
                    assert pt.is_valid_path(new_st)

                    # The shift list corresponds to the stack which is used in
                    # an LR parser, in addition to all the states which are
                    # epsilon transitions. We pop from this list the reduced
                    # path, as long as it matches. Then all popped elements are
                    # replaced by the state that we visit after replaying the
                    # non-terminal reduced by this action.
                    new_sh = prev_sh[:-len(path)] + reduced_path
                    assert pt.is_valid_path(new_sh)

                    # When reducing, we replay terms which got previously
                    # pushed on the stack as our lookahead. These terms are
                    # computed here such that we can traverse the graph from
                    # `to` state, using the replayed terms.
                    new_rp = [nt]
                    if replay > 0:
                        stacked_terms = [ edge.term for edge in path if pt.term_is_stacked(edge.term) ]
                        new_rp = new_rp + stacked_terms[-replay:]
                    new_rp = new_rp + rp
                    new_la = la[:max(len(la) - replay, 0)]

                    # If we are reducing, this implies that we are not
                    # following the edge of the reducing action, and resume the
                    # execution at the last edge of the shift action. At this
                    # point the execution and the stack diverge from standard
                    # LR parser. However, the stack is still manipulated
                    # through Unwind and Replay actions but the state which is
                    # executed no longer matches the last element of the
                    # shifted term or action.
                    if reducing:
                        to = new_sh[-1].src
                    yield APS(to, new_st, new_sh, new_la, new_rp, hs + [edge], reducing=reducing)
            else:
                to = Edge(to, None)
                yield APS(to.src, st, prev_sh + [to], la, rp, hs + [edge])

    def string(self, name = "aps"):
        return """{}.stack = [{}]
{}.shift = [{}]
{}.lookahead = [{}]
{}.replay = [{}]
{}.history = [{}]
        """.format(
            name, " ".join(edge_str(e) for e in self.stack),
            name, " ".join(edge_str(e) for e in self.shift),
            name, ", ".join(repr(e) for e in self.lookahead),
            name, ", ".join(repr(e) for e in self.replay),
            name, " ".join(edge_str(e) for e in self.history)
        )

    def __str__(self):
        return self.string()

def aps_lanes_str(aps_lanes, header = "lanes:", name = "\taps"):
    return "{}\n{}".format(header, "\n".join(aps.string(name) for aps in aps_lanes))


