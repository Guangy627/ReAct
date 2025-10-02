'''
### Prompt templates for WebThink (ReAct style)
just for better review, i formatted the prompt; but in usage, please list all key info in one line.
'''

# System message template for WebThink (ReAct style)
WEBTHINK_SYSTEM_TEMPLATE = """You are an intelligent reasoning agent.
Your job is to determine whether a claim is SUPPORTED, REFUTED, or if there is NOT ENOUGH INFORMATION,
by reasoning step by step and interacting with an external environment.

At each step, you must:
1. Write down your reasoning as a "Thought".
2. Choose an "Action" (Search[], Lookup[], or Finish[]).
3. Observe the result as "Observation".

Repeat this Thought → Action → Observation loop until you can confidently decide Finish[SUPPORTS], Finish[REFUTES], or Finish[NOT ENOUGH INFO].

"""

# Few-shot demonstration for WebThink (as user prompt) 
WEBTHINK_USER_TEMPLATE = """# Task
Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.

# Example 1
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I should search for information about Nikolaj Coster-Waldau and his relation to Fox.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau ... appeared in the 2009 Fox television film Virtuality.
Thought 2: This shows he worked with Fox.
Action 2: Finish[SUPPORTS]

# Example 2
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things and check the setting.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things ... set in the 1980s, primarily in the fictional town of Hawkins, Indiana.
Thought 2: Bloomington does not match Hawkins.
Action 2: Finish[REFUTES]

# Example 3
Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.
Thought 1: I should search for the song Beautiful.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: Beautiful (Christina Aguilera song).
Thought 2: Try searching with more detail.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by Christina Aguilera in 2002.
Thought 3: Now I need to check the Billboard ranking.
Action 3: Lookup[Billboard Hot 100]
Observation 3: The song peaked at number two on the Billboard Hot 100 in the United States.
Thought 4: This confirms the claim.
Action 4: Finish[SUPPORTS]

"""

# Combine into DEFAULT_TEMPLATES dict
DEFAULT_TEMPLATES = {
    "webthink_system_message": WEBTHINK_SYSTEM_TEMPLATE,
    "webthink_user": WEBTHINK_USER_TEMPLATE,
    #...if you have other metadata, add here``
}



###################custom prompt
### Prompt templates for WebThink (ReAct style)
# 在使用时请保持一行，下面只是为了更清晰地展示结构。

# System message template for WebThink (ReAct style)
WEBTHINK_SYSTEM_TEMPLATE = """You are an intelligent reasoning agent.
Your job is to determine whether a claim is SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION,
by reasoning step by step and interacting with an external environment.

At each step, you must:
1. Write down your reasoning as a "Thought".
2. Choose an "Action" from the following set:
   - Search[entity]
   - Lookup[keyword]
   - Finish[SUPPORTS], Finish[REFUTES], or Finish[NOT ENOUGH INFO]
3. Observe the result as "Observation".

Repeat this Thought → Action → Observation loop until you can confidently decide with a Finish action.
"""

# Few-shot demonstration for WebThink (as user prompt) 
WEBTHINK_USER_TEMPLATE = """# Task
Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.

# Example 1 (SUPPORTS)
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I should search for information about Nikolaj Coster-Waldau and Fox.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.
Thought 2: This proves he worked with Fox.
Action 2: Finish[SUPPORTS]

# Example 2 (REFUTES)
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things and check its setting.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is set in the fictional town of Hawkins, Indiana.
Thought 2: Bloomington does not match Hawkins, so the claim is false.
Action 2: Finish[REFUTES]

# Example 3 (NOT ENOUGH INFO)
Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.
Thought 1: I should search for the song Beautiful.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: Beautiful (Christina Aguilera song).
Thought 2: Try searching the Christina Aguilera song.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by Christina Aguilera in 2002.
Thought 3: Now I need to check the Billboard ranking.
Action 3: Lookup[Billboard Hot 100]
Observation 3: The song peaked at number two on the Billboard Hot 100 in the United States.
Thought 4: The ranking is confirmed, but the year 2003 is not mentioned. I cannot be sure.
Action 4: Finish[NOT ENOUGH INFO]
"""

# Combine into DEFAULT_TEMPLATES dict
DEFAULT_TEMPLATES = {
    "webthink_system_message": WEBTHINK_SYSTEM_TEMPLATE,
    "webthink_user": WEBTHINK_USER_TEMPLATE,
}
