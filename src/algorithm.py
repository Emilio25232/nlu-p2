from src.state import State
from src.conllu.conllu_token import Token


class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                                 parser action to be taken in the given state.

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Transition):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition
    

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Extracts features from a given parsing state for use in a transition-based dependency parser.

        This function generates a feature representation from the current state of the parser, 
        which includes features from both the stack and the buffer. The number of features from 
        the stack and the buffer can be specified.

        Parameters:
            nbuffer_feats (int): The number of features to extract from the buffer.
            nstack_feats (int): The number of features to extract from the stack.

        Returns:
            list[str]: A list of extracted features. The features include the words and their 
                    corresponding UPOS (Universal Part-of-Speech) tags from the specified number 
                    of elements in the stack and buffer. The format of the feature list is as follows:
                    [Word_stack_n,...,Word_stack_0, Word_buffer_0,...,Word_buffer_m, 
                        UPOS_stack_n,...,UPOS_stack_0, UPOS_buffer_0,...,UPOS_buffer_m]
                    where 'n' is nstack_feats and 'm' is nbuffer_feats.

        Examples:
            Example 1:
                State: Stack (size=1): (0, ROOT, ROOT_UPOS)
                    Buffer (size=13): (1, Distribution, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=0): []

                Output: ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']

            Example 2:
                State: Stack (size=2): (0, ROOT, ROOT_UPOS) | (1, Distribution, NOUN)
                    Buffer (size=10): (4, license, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=2): [(4, 'det', 3), (4, 'case', 2)]

                Output: ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
        """
        pad_token = "<PAD>"

        # 1. Extract Stack Features (Top 'n' elements)
        # We need the list elements from deep to top (e.g., [ROOT, Word]).
        # If the stack is shorter than n, we pad on the LEFT.
        if nstack_feats > 0 and len(self._state.S) > 0:
            stack_elements = self._state.S[-nstack_feats:]
        else:
            stack_elements = []
        
        missing_stack = nstack_feats - len(stack_elements)

        # 2. Extract Buffer Features (First 'm' elements)
        # We need the first elements of the buffer.
        # If the buffer is shorter than m, we pad on the RIGHT.
        buffer_elements = self._state.B[:nbuffer_feats]
        missing_buffer = nbuffer_feats - len(buffer_elements)

        # 3. Build Word Features List
        word_feats = []
        # Stack Words: Pad first, then actual words (matches Example 1: ['<PAD>', 'ROOT'])
        word_feats.extend([pad_token] * missing_stack)
        word_feats.extend([token.form for token in stack_elements])
        # Buffer Words: Actual words, then Pad
        word_feats.extend([token.form for token in buffer_elements])
        word_feats.extend([pad_token] * missing_buffer)

        # 4. Build UPOS Features List
        pos_feats = []
        # Stack UPOS: Pad first, then actual tags
        pos_feats.extend([pad_token] * missing_stack)
        pos_feats.extend([token.upos for token in stack_elements])
        # Buffer UPOS: Actual tags, then Pad
        pos_feats.extend([token.upos for token in buffer_elements])
        pos_feats.extend([pad_token] * missing_buffer)

        # 5. Combine: [Words + POS]
        return word_feats + pos_feats
    

    def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"



class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        """
        LEFT-ARC is valid iff:
        - Stack is non-empty
        - Buffer is non-empty
        - Top of stack is not ROOT
        - Top of stack does NOT already have a head
        """
        if not state.S:
            return False
        if not state.B:
            return False

        s = state.S[-1]

        # Do not attach ROOT as dependent
        if s.id == 0:
            return False

        # Ensure single-head constraint
        if self.has_head(s, state.A):
            return False

        return True


    def LA_is_correct(self, state: State) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is the correct action for the current parsing state.

        This method checks if applying a LEFT-ARC transition will correctly reflect the dependency
        structure of the sentence being parsed, based on the current state of the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is the correct action in the current state, False otherwise.
        """
        # A LEFT-ARC is correct when:
        # - It is structurally valid
        # - In the gold tree, the buffer head is the head of the stack top
        # - There is no token in the buffer whose gold head is the stack top
        if not self.LA_is_valid(state):
            return False

        top_stack: Token = state.S[-1]
        buffer_head: Token = state.B[0]

        # Gold condition: buffer head is the head of stack top
        if top_stack.head != buffer_head.id:
            return False

        # Do not remove a node that still has (gold) dependents in the buffer
        for tok in state.B:
            if tok.head == top_stack.id:
                return False

        return True
    
    def RA_is_correct(self, state: State) -> bool:
        """
        Determines if a RIGHT-ARC (RA) transition is the correct action for the current parsing state.

        This method assesses whether applying a RIGHT-ARC transition aligns with the correct 
        dependency structure of the sentence, based on the parser's current state.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a RIGHT-ARC transition is the correct action in the current state, False otherwise.
        """
        # A RIGHT-ARC is correct when:
        # - It is structurally valid
        # - In the gold tree, the stack top is the head of the buffer head
        if not self.RA_is_valid(state):
            return False

        top_stack: Token = state.S[-1]
        buffer_head: Token = state.B[0]

        return buffer_head.head == top_stack.id

    def RA_is_valid(self, state: State) -> bool:
        """
        RIGHT-ARC is valid iff:
        - Stack is non-empty
        - Buffer is non-empty
        - Buffer head does NOT already have a head
        """
        if not state.S:
            return False
        if not state.B:
            return False

        b = state.B[0]

        # Ensure single-head constraint for buffer head
        if self.has_head(b, state.A):
            return False

        return True

    def REDUCE_is_correct(self, state: State) -> bool:
        """
        Determines if applying a REDUCE transition is the correct action for the current parsing state.

        A REDUCE transition is correct if there is no word in the buffer (state.B) whose head 
        is the word on the top of the stack (state.S[-1]). This method checks this condition 
        against the current state of the parser.

        REDUCE can be only correct iff for every word in the buffer, 
        no word has as head the top word from the stack 

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is the correct action in the current state, False otherwise.
        """
        # It is correct to REDUCE when:
        # - It is structurally valid (token on top of the stack already has a head)
        # - There is no token in the buffer whose gold head is the stack-top token
        if not self.REDUCE_is_valid(state):
            return False

        top_stack: Token = state.S[-1]
        for tok in state.B:
            if tok.head == top_stack.id:
                return False

        return True

    def REDUCE_is_valid(self, state: State) -> bool:
        """
        REDUCE is valid iff:
        - Stack is non-empty
        - Top of stack already has a head
        """
        if not state.S:
            return False

        s = state.S[-1]

        # Only reduce if token already has a head
        return self.has_head(s, state.A)

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step, given an input dependency tree.

        This function iterates through a given sentence, represented as a dependency tree, to generate a sequence 
        of gold-standard transitions. These transitions are what an ideal parser should predict at each step to 
        correctly parse the sentence. The function checks the validity and correctness of possible transitions 
        at each step and selects the appropriate one based on the arc-eager parsing algorithm. It is primarily 
        used for later training a dependency parser.

        Parameters:
            sent (list['Token']): A list of 'Token' instances representing a dependency tree. Each 'Token' 
                        should contain information about a word/token in a sentence.

        Returns:
            samples (list['Sample']): A list of Sample instances. Each Sample stores an state instance and a transition instance
            with the information of the outputs to predict (the transition and optionally the dependency label)
        """

        state = self.create_initial_state(sent) 
        samples = [] #Store here all training samples for sent
        
        # Pre-compute gold arcs for assertion at the end
        gold_arcs_set = self.gold_arcs(sent)

        #Applies the transition system until a final configuration state is reached
        while not self.final_state(state):
            
            # Create a snapshot of the state for the sample
            current_state_snapshot = State(list(state.S), list(state.B), set(state.A))

            if self.LA_is_valid(state) and self.LA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the LA transition using the function apply_transition
                # LEFT-ARC: buffer head is the head of stack top
                top_stack: Token = state.S[-1]
                buffer_head: Token = state.B[0]
                # Gold label is the deprel of the dependent (stack top)
                transition = Transition(self.LA, top_stack.dep)
                # Store a copy of the current state and the chosen transition
                samples.append(Sample(State(list(state.S), list(state.B), set(state.A)), transition))
                # Update state
                self.apply_transition(state, transition)

            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the RA transition using the function apply_transition
                # RIGHT-ARC: stack top is the head of buffer head
                buffer_head: Token = state.B[0]
                # Gold label is the deprel of the dependent (buffer head)
                transition = Transition(self.RA, buffer_head.dep)
                samples.append(Sample(State(list(state.S), list(state.B), set(state.A)), transition))
                self.apply_transition(state, transition)

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the REDUCE transition using the function apply_transition
                transition = Transition(self.REDUCE)
                samples.append(Sample(State(list(state.S), list(state.B), set(state.A)), transition))
                self.apply_transition(state, transition)
            else:
                #If no other transiton can be applied, it's a SHIFT transition
                transition = Transition(self.SHIFT)
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                samples.append(Sample(current_state_snapshot, transition))
                #Update the state by applying the SHIFT transition using the function apply_transition
                self.apply_transition(state,transition)


        #When the oracle ends, the generated arcs must
        #match exactly the gold arcs, otherwise the obtained sequence of transitions is not correct
        assert gold_arcs_set == state.A, f"Gold arcs {gold_arcs_set} and generated arcs {state.A} do not match"
    
        return samples   

    
    def has_head(self, token: 'Token', arcs: set) -> bool:
        """
        Checks if the token already has a head assigned (appears as a dependent in an arc).
        
        Args:
            token: The Token to check.
            arcs: A set of arcs in the format (head_id, dependency_label, dependent_id).
        
        Returns:
            True if the token appears as a dependent (3rd element) in any arc.
        """
        if token is None:
            return False
        # Check if the token ID appears as the dependent (index 2) in any existing arc
        return any(arc[2] == token.id for arc in arcs)
      
    

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies a given Arc-Eager transition to the current parsing state.
        
        Args:
            state: The current parsing state (Stack, Buffer, Arcs).
            transition: The transition to apply (action, dependency).
        """
        t = transition.action
        dep = transition.dependency
        
        # Get top of Stack (s) and top of Buffer (b) safely
        s = state.S[-1] if state.S else None  
        b = state.B[0] if state.B else None   

        if t == self.LA and self.LA_is_valid(state):
            # LEFT-ARC: Creates an arc where the Head is the Buffer top (b) and Dependent is the Stack top (s).
            # Then, removes the element from the Stack.
            state.A.add((b.id, dep, s.id))
            state.S.pop()

        elif t == self.RA and self.RA_is_valid(state): 
            # RIGHT-ARC: Creates an arc where the Head is the Stack top (s) and Dependent is the Buffer top (b).
            # Then, moves the element from the Buffer to the Stack.
            state.A.add((s.id, dep, b.id))
            state.S.append(b)
            del state.B[0]

        elif t == self.REDUCE and self.has_head(s, state.A): 
            # REDUCE: Removes the element from the Stack because it already has a head assigned.
            state.S.pop()

        else:
            # SHIFT: Moves the element from the Buffer to the Stack.
            # (Default fallback action)
            state.S.append(b) 
            del state.B[:1]
    


    def gold_arcs(self, sent: list['Token']) -> set:
        """
        Extracts and returns the gold-standard dependency arcs from a given sentence.

        This function processes a sentence represented by a list of Token objects to extract the dependency relations 
        (arcs) present in the sentence. Each Token object should contain information about its head (the id of the 
        parent token in the dependency tree), the type of dependency, and its own id. The function constructs a set 
        of tuples, each representing a dependency arc in the sentence.

        Parameters:
            sent (list[Token]): A list of Token objects representing the sentence. Each Token object contains 
                                information about a word or punctuation in a sentence, including its dependency 
                                relations and other annotations.

        Returns:
            gold_arcs (set[tuple]): A set of tuples, where each tuple is a triplet (head_id, dependency_type, dependent_id). 
                                    This represents all the gold-standard dependency arcs in the sentence. The head_id and 
                                    dependent_id are integers representing the respective tokens in the sentence, and 
                                    dependency_type is a string indicating the type of dependency relation.
        """
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))

        return gold_arcs


   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)

    print()
    print("**************************************************")
    print("*     Testing Oracle                             *")
    print("**************************************************")
    
    print("Running oracle on the example sentence...")
    samples = arc_eager.oracle(tree)
    print(f"Generated {len(samples)} samples.")
    for i, sample in enumerate(samples):
        print(f"Step {i+1}: {sample.transition}")
        # print(sample.state) # Optional: print state to debug
    
    print("\nOracle test passed!")