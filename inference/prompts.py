"""
Biomedical domain prompts for Think-on-Graph over PrimeKG.
Mirrors the structure of ToG/prompt_list.py but with biomedical few-shot examples.
"""

# ============================================================
# Entity Extraction — NEW (STaRK QA has no pre-annotated topic entities)
# ============================================================

extract_entities_prompt_bio = """Given a biomedical question, extract the key entity names that should be looked up in a biomedical knowledge graph. Return ONLY a JSON list of entity name strings, nothing else.

The knowledge graph contains these types of entities: disease, gene/protein, drug, molecular_function, pathway, anatomy, effect/phenotype, biological_process, cellular_component, exposure.

Q: Which drugs are indicated for treating Alzheimer's disease and also target the BACE1 protein?
A: ["Alzheimer disease", "BACE1"]

Q: What are the known side effects of Metformin?
A: ["Metformin"]

Q: Which genes are associated with breast cancer and involved in the apoptosis pathway?
A: ["breast cancer", "apoptosis"]

Q: What proteins interact with TP53 and are expressed in the liver?
A: ["TP53", "liver"]

Q: %s
A: """

# ============================================================
# Relation Pruning
# ============================================================

extract_relation_prompt_bio = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Which drugs are indicated for treating Alzheimer's disease?
Topic Entity: Alzheimer disease (disease)
Relations:
1. associated with
2. contraindication
3. indication
4. linked to
5. parent-child
6. phenotype present
A: 1. {indication (Score: 0.6)}: This relation directly connects diseases to drugs indicated for treating them, which is central to finding drugs used to treat Alzheimer's disease.
2. {associated with (Score: 0.3)}: This relation can reveal gene/protein associations related to Alzheimer's disease, helping identify potential therapeutic targets.
3. {linked to (Score: 0.1)}: This relation may provide additional contextual links between entities in the biomedical domain.

Q: What are the known side effects of Metformin?
Topic Entity: Metformin (drug)
Relations:
1. carrier
2. contraindication
3. enzyme
4. indication
5. off-label use
6. side effect
7. synergistic interaction
8. target
9. transporter
A: 1. {side effect (Score: 0.8)}: This relation directly provides the side effects associated with Metformin, which is exactly what the question asks.
2. {contraindication (Score: 0.1)}: Contraindications may overlap with side effects in some clinical contexts.
3. {target (Score: 0.1)}: Drug targets can sometimes explain mechanism-related side effects.

Q: """

# ============================================================
# Entity Scoring
# ============================================================

score_entity_candidates_prompt_bio = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: Which drugs target the ACE2 protein?
Relation: target
Entities: Captopril; Lisinopril; Remdesivir; Aspirin
Score: 0.35, 0.35, 0.2, 0.1
ACE2 (Angiotensin-converting enzyme 2) is a known target of ACE inhibitors like Captopril and Lisinopril. Remdesivir has some indirect interaction, while Aspirin is less directly relevant.

Q: {}
Relation: {}
Entities: """

# ============================================================
# Reasoning Check (sufficiency)
# ============================================================

prompt_evaluate_bio = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Which drugs are indicated for treating Type 2 diabetes?
Knowledge Triplets: Type 2 diabetes, indication, Metformin
Type 2 diabetes, indication, Glipizide
Type 2 diabetes, indication, Sitagliptin
A: {Yes}. Based on the given knowledge triplets, several drugs indicated for treating Type 2 diabetes are identified: Metformin, Glipizide, and Sitagliptin. This is sufficient to answer the question.

Q: What proteins interact with BRCA1 and are also associated with ovarian cancer?
Knowledge Triplets: BRCA1, ppi, BARD1
BRCA1, ppi, RAD51
BRCA1, ppi, TP53
A: {No}. The knowledge triplets show proteins that interact with BRCA1 (BARD1, RAD51, TP53), but we don't yet know which of these are associated with ovarian cancer. Additional exploration is needed.

Q: What are the side effects of Ibuprofen?
Knowledge Triplets: Ibuprofen, side effect, Nausea
Ibuprofen, side effect, Headache
Ibuprofen, side effect, Gastrointestinal bleeding
Ibuprofen, target, Cyclooxygenase 2
A: {Yes}. The knowledge triplets provide several side effects of Ibuprofen including Nausea, Headache, and Gastrointestinal bleeding. This is sufficient to answer the question.

Q: Which genes are associated with Parkinson's disease and involved in dopamine signaling?
Knowledge Triplets: Parkinson's disease, associated with, LRRK2
Parkinson's disease, associated with, SNCA
Parkinson's disease, associated with, PARK7
LRRK2, associated with, dopamine signaling
A: {Yes}. The triplets show genes associated with Parkinson's disease (LRRK2, SNCA, PARK7), and LRRK2 is also associated with dopamine signaling. This provides a sufficient basis to answer the question.

"""

# ============================================================
# Answer Generation
# ============================================================

answer_prompt_bio = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your own knowledge.
IMPORTANT: You MUST wrap your final answer entity name(s) in curly braces like {answer}. Do not omit the curly braces.

Q: Which drugs are indicated for treating Type 2 diabetes?
Knowledge Triplets: Type 2 diabetes, indication, Metformin
Type 2 diabetes, indication, Glipizide
Type 2 diabetes, indication, Sitagliptin
A: Based on the given knowledge triplets, drugs indicated for treating Type 2 diabetes include Metformin, Glipizide, and Sitagliptin. Therefore, the answer is {Metformin, Glipizide, Sitagliptin}.

Q: What proteins interact with BRCA1 and are also associated with ovarian cancer?
Knowledge Triplets: BRCA1, ppi, BARD1
BRCA1, ppi, RAD51
BRCA1, ppi, TP53
TP53, associated with, Ovarian cancer
BARD1, associated with, Ovarian cancer
A: Based on the given knowledge triplets, BRCA1 interacts with BARD1, RAD51, and TP53. Among these, TP53 and BARD1 are associated with ovarian cancer. Therefore, the answer is {TP53, BARD1}.

Q: What are the side effects of Ibuprofen?
Knowledge Triplets: Ibuprofen, side effect, Nausea
Ibuprofen, side effect, Headache
Ibuprofen, side effect, Gastrointestinal bleeding
A: Based on the given knowledge triplets, the known side effects of Ibuprofen include Nausea, Headache, and Gastrointestinal bleeding. Therefore, the answer is {Nausea, Headache, Gastrointestinal bleeding}.

Q: %s
"""

# ============================================================
# Chain-of-Thought (CoT) fallback
# ============================================================

cot_prompt_bio = """Answer the following biomedical question step by step. IMPORTANT: Wrap your final answer entity name(s) in curly braces like {answer}.

Q: What is the mechanism of action of Metformin in treating Type 2 diabetes?
A: First, Metformin is an oral antidiabetic drug. Second, it primarily works by decreasing hepatic glucose production and increasing insulin sensitivity in peripheral tissues. The answer is {decreasing hepatic glucose production and increasing insulin sensitivity}.

Q: Which gene mutations are most commonly associated with hereditary breast cancer?
A: First, hereditary breast cancer is strongly associated with BRCA1 and BRCA2 gene mutations. Second, these tumor suppressor genes play critical roles in DNA repair. The answer is {BRCA1, BRCA2}.

Q: What is the primary anatomical site affected by Crohn's disease?
A: First, Crohn's disease is a type of inflammatory bowel disease. Second, it most commonly affects the terminal ileum and colon. The answer is {terminal ileum and colon}.

Q: Which drug class is primarily used as first-line treatment for hypertension?
A: First, hypertension treatment guidelines recommend several first-line drug classes. Second, ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics are all considered first-line options. The answer is {ACE inhibitors, ARBs, calcium channel blockers, thiazide diuretics}."""


# ============================================================
# Dynamic Subquery Generation
# ============================================================

# At each hop, given the original query, the gold evidence path,
# and the current node, the LLM must produce a focused sub-question
# (and optionally the type of the next node we should look for).

dynamic_subquery_prompt_with_type = """You are guiding a multi-hop knowledge graph path-finder for biomedical question answering.

Original question: {query}

Gold evidence path (the correct chain of facts that answers the question):
{evidence_path}

Current state of path-finding:
- We are currently at node "{current_name}" of type "{current_type}".
- This is hop {hop_idx} of {total_hops}.

Your task:
Generate ONE focused sub-question that the path-finder should ask at THIS step in order to make progress toward the answer. The sub-question should help select the next entity along the evidence path. Also output the entity type that we should look for in the next hop.

Output JSON only, no commentary, no markdown:
{{"subquery": "<your sub-question>", "target_type": "<expected node type>"}}
"""


dynamic_subquery_prompt_no_type = """You are guiding a multi-hop knowledge graph path-finder for biomedical question answering.

Original question: {query}

Gold evidence path (the correct chain of facts that answers the question):
{evidence_path}

Current state of path-finding:
- We are currently at node "{current_name}" of type "{current_type}".
- This is hop {hop_idx} of {total_hops}.

Your task:
Generate ONE focused sub-question that the path-finder should ask at THIS step in order to make progress toward the answer. The sub-question should help select the next entity along the evidence path.

Output JSON only, no commentary, no markdown:
{{"subquery": "<your sub-question>"}}
"""
