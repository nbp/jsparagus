import itertools
from .ordered import OrderedFrozenSet
from .grammar import InitNt, Nt, Some

class Action:
    __slots__ = [
        "read",    # Set of trait names which are consumed by this action.
        "write",   # Set of trait names which are mutated by this action.
        "_hash",   # Cached hash.
    ]

    def __init__(self, read, write):
        assert isinstance(read, list)
        assert isinstance(write, list)
        self.read = read
        self.write = write
        self._hash = None

    def is_inconsistent(self):
        """Returns True whether this action is inconsistent. An action can be
        inconsistent if the parameters it is given cannot be evaluated given
        its current location in the parse table. Such as CheckNotOnNewLine.
        """
        return False

    def is_condition(self):
        "Unordered condition, which accept or not to reach the next state."
        return False

    def condition(self):
        "Return the conditional action."
        raise TypeError("Action::condition not implemented")

    def check_same_variable(self, other):
        "Return whether both conditional are checking the same variable."
        raise TypeError("Action::check_same_variable not implemented")

    def check_different_values(self, other):
        "Return whether both conditional are checking non-overlapping values."
        raise TypeError("Action::check_different_values not implemented")

    def follow_edge(self):
        """Whether the execution of this action resume following the epsilon transition
        (True) or if it breaks the graph epsilon transition (False) and returns
        at a different location, defined by the top of the stack."""
        return True

    def update_stack(self):
        """Whether the execution of this action changes the parser stack."""
        return False

    def update_stack_with(self):
        """Returns a tuple which represents the mutation to be applied to the parser
        stack. This tuple is composed of 3 elements, the first one is the
        number of popped terms, the second is the forking terms, and
        the third is the number of replayed terms.

        The forking term can either be an ErrorToken or a non-terminal. It is
        named the forking term as this is the term which change the path taken
        after replaying the terms which are on the stack. Especially when there
        is no pop-ed elements. """
        raise TypeError("Action::update_stack_with not implemented")

    def shifted_action(self, shifted_term):
        "Returns the same action shifted by a given amount."
        return self

    def contains_accept(self):
        "Returns whether the current action stops the parser."
        return False

    def rewrite_state_indexes(self, state_map):
        """If the action contains any state index, use the map to map the old index to
        the new indexes"""
        return self

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.read) != sorted(other.read):
            return False
        if sorted(self.write) != sorted(other.write):
            return False
        for s in self.__slots__:
            if getattr(self, s) != getattr(other, s):
                return False
        return True

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        def hashed_content():
            yield self.__class__
            yield "rd"
            for alias in self.read:
                yield alias
            yield "wd"
            for alias in self.write:
                yield alias
            for s in self.__slots__:
                yield repr(getattr(self, s))
        self._hash = hash(tuple(hashed_content()))
        return self._hash

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __repr__(self):
        return str(self)

class Replay(Action):
    """Replay a term which was previously saved by the Unwind function. Note that
    this does not Shift a term given as argument as the replay action should
    always be garanteed and that we want to maximize the sharing of code when
    possible."""
    __slots__ = "replay_dest",

    def __init__(self, replay_dest):
        super().__init__([], [])
        assert isinstance(replay_dest, int)
        self.replay_dest = replay_dest

    def __str__(self):
        return "Replay({})".format(str(replay_dest))

    def update_stack(self):
        return True

    def update_stack_with(self):
        return (0, None, -1)

    def rewrite_state_indexes(self, state_map):
        return Replay(state_map[self.replay_dest])

class Unwind(Action):
    """Define an unwind operation which pops N elements of the stack and pushes one
    non-terminal."""
    __slots__ = 'nt', 'replay', 'pop'

    def __init__(self, nt, pop, replay = 0):
        name = nt.name
        if isinstance(name, InitNt):
            name = "Start_" + name.goal.name
        super().__init__([], ["nt_" + name])
        self.nt = nt    # Non-terminal which is reduced
        self.pop = pop  # Number of stack elements which should be replayed.
        self.replay = replay # List of terms to shift back

    def __str__(self):
        return "Unwind({}, {}, {})".format(self.nt, self.pop, self.replay)

    def update_stack(self):
        return True

    def update_stack_with(self):
        return (self.pop, self.nt, self.replay)

    def shifted_action(self, shifted_term):
        return Unwind(self.nt, self.pop, replay = self.replay + 1)

class Reduce(Action):
    """Prevent the fall-through to the epsilon transition and returns to the shift
    table execution to resume shifting or replaying terms."""
    __slots__ = 'unwind',

    def __init__(self, unwind):
        assert isinstance(unwind, Unwind)
        assert not unwind.is_condition()
        assert not unwind.is_inconsistent()
        assert not unwind.contains_accept()
        super().__init__(unwind.read, unwind.write)
        self.unwind = unwind

    def __str__(self):
        return "Reduce({})".format(str(self.unwind))

    def update_stack(self):
        return self.unwind

    def update_stack_with(self):
        return self.unwind.update_stack_with()

    def follow_edge(self):
        return False

    def shifted_action(self, shifted_term):
        unwind = self.unwind.shifted_action(shifted_term)
        return Reduce(unwind)

class Accept(Action):
    """This state terminate the parser by accepting the content consumed until
    now."""
    __slots__ = ()

    def __init__(self):
        super().__init__([], [])

    def __str__(self):
        return "Accept()"

    def contains_accept(self):
        "Returns whether the current action stops the parser."
        return True

    def shifted_action(self, shifted_term):
        return Accept()

class Lookahead(Action):
    """Define a Lookahead assertion which is meant to either accept or reject
    sequences of terminal/non-terminals sequences."""
    __slots__ = 'terms', 'accept'

    def __init__(self, terms, accept):
        assert isinstance(terms, (OrderedFrozenSet, frozenset))
        assert all(not isinstance(t, Nt) for t in terms)
        assert isinstance(accept, bool)
        super().__init__([], [])
        self.terms = terms
        self.accept = accept

    def is_inconsistent(self):
        # A lookahead restriction cannot be encoded in code, it has to be
        # solved using fix_with_lookahead.
        return True

    def is_condition(self):
        return True

    def condition(self):
        return self

    def check_same_variable(self, other):
        raise TypeError("Lookahead::check_same_variables: Lookahead are always inconsistent")

    def check_different_values(self, other):
        raise TypeError("Lookahead::check_different_values: Lookahead are always inconsistent")

    def __str__(self):
        return "Lookahead({}, {})".format(self.terms, self.accept)

    def shifted_action(self, shifted_term):
        if isinstance(shifted_term, Nt):
            return True
        if shifted_term in self.terms:
            return self.accept
        return not self.accept

class CheckNotOnNewLine(Action):
    """Check whether the terminal at the given stack offset is on a new line or
    not. If not this would produce an Error, otherwise this rule would be
    shifted."""
    __slots__ = 'offset',

    def __init__(self, offset = 0):
        # assert offset >= -1 and "Smaller offsets are not supported on all backends."
        super().__init__([], [])
        self.offset = offset

    def is_inconsistent(self):
        # We can only look at stacked terminals. Having an offset of 0 implies
        # that we are looking for the next terminal, which is not yet shifted.
        # Therefore this action is inconsistent as long as the terminal is not
        # on the stack.
        return self.offset >= 0

    def is_condition(self):
        return True

    def condition(self):
        return self

    def check_same_variable(self, other):
        return self.offset == other.offset

    def check_different_values(self, other):
        return False

    def shifted_action(self, shifted_term):
        if isinstance(shifted_term, Nt):
            return True
        return CheckNotOnNewLine(self.offset - 1)

    def __str__(self):
        return "CheckNotOnNewLine({})".format(self.offset)

class FilterStates(Action):
    """Check whether the stack at the top matches the state value, if so transition
    to the destination, otherwise check other states."""
    __slots__ = 'states',

    def __init__(self, states):
        assert isinstance(states, (list, tuple, OrderedFrozenSet))
        super().__init__([], [])
        # Set of states which can follow this transition.
        self.states = OrderedFrozenSet(sorted(states))

    def is_condition(self):
        return True

    def condition(self):
        return self

    def check_same_variable(self, other):
        return isinstance(other, FilterStates)

    def check_different_values(self, other):
        assert isinstance(other, FilterStates)
        return self.states.isdisjoint(other.states)

    def rewrite_state_indexes(self, state_map):
        """If the action contains any state index, use the map to map the old index to
        the new indexes"""
        states = list(state_map[s] for s in self.states)
        return FilterStates(states)

    def __str__(self):
        return "FilterStates({})".format(self.states)

class FilterFlag(Action):
    """Define a filter which check for one value of the flag, and continue to the
    next state if the top of the flag stack matches the expected value."""
    __slots__ = 'flag', 'value'

    def __init__(self, flag, value):
        super().__init__(["flag_" + flag], [])
        self.flag = flag
        self.value = value

    def is_condition(self):
        return True

    def condition(self):
        return self

    def check_same_variable(self, other):
        return isinstance(other, FilterFlag) and self.flag == other.flag

    def check_different_values(self, other):
        assert isinstance(other, FilterFlag)
        return self.value != other.value

    def __str__(self):
        return "FilterFlag({}, {})".format(self.flag, self.value)

class PushFlag(Action):
    """Define an action which pushes a value on a stack dedicated to the flag. This
    other stack correspond to another parse stack which live next to the
    default state machine and is popped by PopFlag, as-if this was another
    reduce action. This is particularly useful to raise the parse table from a
    LR(0) to an LR(k) without needing as much state duplications."""
    __slots__ = 'flag', 'value'

    def __init__(self, flag, value):
        super().__init__([], ["flag_" + flag])
        self.flag = flag
        self.value = value

    def __str__(self):
        return "PushFlag({}, {})".format(self.flag, self.value)

class PopFlag(Action):
    """Define an action which pops a flag from the flag bit stack."""
    __slots__ = 'flag',

    def __init__(self, flag):
        super().__init__(["flag_" + flag], ["flag_" + flag])
        self.flag = flag

    def __str__(self):
        return "PopFlag({})".format(self.flag)

class FunCall(Action):
    """Define a call method operation which reads N elements of he stack and
    pushpathne non-terminal. The replay attribute of a reduce action correspond
    to the number of stack elements which would have to be popped and pushed
    again using the parser table after reducing this operation. """
    __slots__ = 'trait', 'method', 'offset', 'args', 'fallible', 'set_to'

    def __init__(self, method, args,
                 trait = "AstBuilder",
                 fallible = False,
                 set_to = "val",
                 offset = 0,
                 alias_read = [],
                 alias_write = []):
        super().__init__(alias_read, alias_write)
        self.trait = trait       # Trait on which this method is implemented.
        self.method = method     # Method and argument to be read for calling it.
        self.fallible = fallible # Whether the function call can fail.
        self.offset = offset     # Offset to add to each argument offset.
        self.args = args         # Tuple of arguments offsets.
        self.set_to = set_to     # Temporary variable name to set with the result.

    def __str__(self):
        return "{} = {}::{}({}){} [off: {}]".format(
            self.set_to, self.trait, self.method,
            ", ".join(map(str, self.args)),
            self.fallible and '?' or '',
            self.offset)

    def __repr__(self):
        return "FunCall({})".format(', '.join(map(repr, [
            self.trait, self.method, self.fallible, self.read, self.write,
            self.args, self.set_to, self.offset
        ])))

    def map_args(self, f):
        return FunCall(self.method, tuple(f(self.offset, a) for a in self.args),
                       trait = self.trait,
                       fallible = self.fallible,
                       set_to = self.set_to,
                       offset = 0,
                       alias_read = self.read,
                       alias_write = self.write)

    def replace_set_to(self, name):
        return FunCall(self.method, self.args,
                       trait = self.trait,
                       fallible = self.fallible,
                       set_to = name,
                       offset = self.offset,
                       alias_read = self.read,
                       alias_write = self.write)

    def shifted_action(self, shifted_term):
        return FunCall(self.method, self.args,
                       trait = self.trait,
                       fallible = self.fallible,
                       set_to = self.set_to,
                       offset = self.offset + 1,
                       alias_read = self.read,
                       alias_write = self.write)

class Seq(Action):
    """Aggregate multiple actions in one sequence. Note, that the aggregated
    actions should not contain any condition or action which are mutating the
    state. Only the last action aggregated can update the parser stack"""
    __slots__ = 'actions',

    def __init__(self, actions):
        assert isinstance(actions, list)
        read = [ rd for a in actions for rd in a.read ]
        write = [ wr for a in actions for wr in a.write ]
        super().__init__(read, write)
        self.actions = tuple(actions)   # Ordered list of actions to execute.
        assert all(not a.is_condition() for a in actions)
        assert all(not isinstance(a, Seq) for a in actions)
        assert all(a.follow_edge() for a in actions[:-1])
        assert all(not a.update_stack() for a in actions[:-1])

    def __str__(self):
        return "{{ {} }}".format("; ".join(map(str, self.actions)))

    def __repr__(self):
        return "Seq({})".format(repr(self.actions))

    def follow_edge(self):
        return self.actions[-1].follow_edge()

    def update_stack(self):
        return self.actions[-1].update_stack()

    def update_stack_with(self):
        return self.actions[-1].update_stack_with()

    def contains_accept(self):
        return any(a.contains_accept() for a in self.actions)

    def shifted_action(self, shift):
        actions = list(map(lambda a: a.shifted_action(shift), self.actions))
        return Seq(actions)

    def rewrite_state_indexes(self, state_map):
        actions = list(map(lambda a: a.rewrite_state_indexes(state_map), self.actions))
        return Seq(actions)

class SeqBuilder:
    """Aggregate multiple actions in one sequence. Reduce actions added to this
    sequence are implicitly removed except for the last one but are still
    considered for rewriting arguments of FunCall. """

    def __init__(self):
        # offset_stack is used for rewritting offsets which are matching the
        # value of non-terminals.
        self.offset_stack = [1] # list(reversed(range(1, 16)))
        self.popped = 0
        self.replay = []
        # Until a Reduce action is seen, actions are stashed in the
        # stashed_actions list.
        self.actions = []
        self.stashed_actions = []
        self.last_reduce = None

    def ensure_stack(self, n):
        # Always add margin such that the first element is always an integer.
        n = n + 2 - len(self.offset_stack)
        if n < 0:
            return
        base = self.offset_stack[0] + 1
        self.offset_stack = list(itertools.chain(reversed(range(base, base + n)), self.offset_stack))

    def args_rewrite(self, offset, a):
        if isinstance(a, int):
            a = a + offset
            self.ensure_stack(a)
            # Return the variable name associated with the value pushed by
            # the shifted token.
            return self.offset_stack[-a]
        if isinstance(a, str):
            return a
        if isinstance(a, Some):
            return Some(self.args_rewrite(offset, a.inner))
        return a

    def can_shift(self, term):
        if not isinstance(term, Action):
            return len(self.replay) > 0
        assert not term.contains_accept()
        return not term.is_condition() or self.actions == []

    def shift(self, term):
        assert self.can_shift(term)
        if not isinstance(term, Action):
            # Push the token value of the lookahead, as a stack offset.
            self.offset_stack.append(self.replay.pop())
            self.popped -= 1
        elif isinstance(term, FunCall):
            term = term.map_args(self.args_rewrite)
            self.stashed_actions.append(term)
        elif isinstance(term, Reduce):
            t_pop, t_nt, t_replay = term.update_stack_with()
            self.ensure_stack(t_replay + t_pop)
            # Note, if we consume a reduce action before consuming all the
            # lookahead of previous reduce actions, the final reduce should
            # still have the remaining number of lookahead terms to replay.
            replay = t_replay
            while replay > 0:
                self.replay.append(self.offset_stack.pop())
                self.popped += 1
                replay -= 1

            # Remove all the terms pop-ed by all reduce actions seen until now.
            pop = t_pop
            while pop > 0:
                self.offset_stack.pop()
                self.popped += 1
                pop -= 1

            # New reduce action to be added if this is the last one.
            self.last_reduce = Reduce(Unwind(t_nt, self.popped - len(self.replay), len(self.replay)))

            # Move stashed actions to the list of actions to appear in this
            # sequence. The reason being that we end with a reduce action.
            # Therefore as we do not want code to be executed twice, we do not
            # want to append FunCall coming from the path where we reduced
            # into.
            self.actions.extend(self.stashed_actions)
            self.stashed_actions = []

            # The value of the non-terminal pushed on the stack after a
            # replayed action is going to be stored in the "value" variable. We
            # give it another name.
            name = "action" + str(len(self.actions))
            if len(self.actions) >= 1 and isinstance(self.actions[-1], FunCall):
                self.replay.append(name)
                self.actions[-1] = self.actions[-1].replace_set_to(name)
            else:
                self.replay.append(len(self.replay) + 1)

        elif isinstance(term, Seq):
            for act in term.actions:
                self.shift(act)
        else:
            raise ValueError("Unsupported Action: {}".format(str(term)))

    def finish(self):
        if self.last_reduce is not None:
            if isinstance(self.actions[-1], FunCall):
                self.actions[-1] = self.actions[-1].replace_set_to("value")
            self.actions.append(self.last_reduce)
        if len(self.actions) == 1:
            return self.actions[0]
        return Seq(self.actions)
