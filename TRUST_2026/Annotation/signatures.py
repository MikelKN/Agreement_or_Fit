import dspy

# =========================================================================================================================
# EXPERIMENT 1: Preliminary annotation signatures for sentiment, identity targeting, and specific identity group targeted. 
# =========================================================================================================================

class SentimentSignature(dspy.Signature):# Step 1 of labeling scheme. 
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content.
    How would you describe the sentiment of the text: 
    0=Strongly positive, 1=Somewhat positive,  2=Neutral, 3=Somewhat negative, 4=Strongly negative.
    Focus on the overall emotional tone of the text, not sentiment toward any particular person or group.
    """
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="0-4 rating of general sentiment")
    reasoning = dspy.OutputField(desc="1 sentence explaining the rating")

class IdentityTargetDetection(dspy.Signature):
    """
    Does text target a protected identity group?
    protected groups include: race/ethnicity, religion, national origin or citizenship status, gender, sexual orientation, age, disability status.
    IMPORTANT: You must output 'Yes' if the group is mentioned in ANY context. This includes neutral references, positive or supportive messaging, self-identification, and counterspeech defending the group against hate.
    """
    text = dspy.InputField()
    targets_protected_group = dspy.OutputField(desc="Yes or no") 
    explanation = dspy.OutputField(desc="1-2 sentence justification")

class SubIdentitySpecification(dspy.Signature):
    """
    Specify the sub-identity group(s). Format: category:subgroup. 
    Categories/subgroups include:
    race_ethnicity: Black or African American, Latino or non-white Hispanic, Asian, Middle Eastern, Native American or Alaska Native, Pacific Islander, Non-hispanic white
    religion: Jews, Christians, Buddhists, Hindus, Mormons, Atheists, Muslims
    national_origin: [country name], immigrant, migrant worker, undocumented person
    gender: Women, Men, Non-binary or third gender, Transgender women, Transgender men, Transgender (unspecified)
    sexual_orientation: Bisexual, Gay, Lesbian, Heterosexual
    age: Children (0-12), Adolescents (13-17), Young adults (18-39), Middle-aged (40-64), Seniors (65+)
    disability: Physical, Cognitive/learning, Mental health problems, Visually impaired, Hearing impaired
    Examples: "race_ethnicity:Black or African American", "race_ethnicity:Black or African American, Asian",  "religion:Muslims, gender:Women"
    """
    text = dspy.InputField()
    targeted_identities = dspy.OutputField(desc= "category:subgroup format")
    explanation = dspy.OutputField(desc="2-3 sentences justification of which groups are targeted and how")

# =========================================================================================================================
# EXPERIMENT 2: Creating 8 signature for the nine separate dumensions experiemnt(modular)
# =========================================================================================================================
class RespectSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Is this comment respectful towards the identified group(s)? 
    0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful. 
    """
    text = dspy.InputField()
    targeted_identities = dspy.InputField(desc="Target group(s)")
    respect = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class InsultSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 
    0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField()   
    targeted_identities = dspy.InputField(desc="Target group(s)")
    insult = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class HumiliateSignature(dspy.Signature): ##Does text aim to humiliate the identified group 63.2
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 
    0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField()    
    targeted_identities = dspy.InputField(desc="Target group(s)")
    humiliate = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class StatusSignature(dspy.Signature): #Focus on the relative social position the text assigns to the group. 39.6
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    This comment states that the group(s) you previously identified is:? 
    0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.
    """
    text = dspy.InputField()   
    targeted_identities = dspy.InputField(desc="Target group(s)")
    status = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class DehumanizeSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 
    0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)")
    dehumanize = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class ViolenceSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 
    0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)") 
    violence = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class GenocideSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 
    0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)")
    genocide = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class AttackDefendSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Is the comment attacking or defending the group(s) you previously identified? 
    0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.
    
    Distinguish between:
    - Criticism of behavior vs. attack on identity
    - Defending group vs. defending individual members
    """
    text = dspy.InputField()  
    targeted_identities = dspy.InputField(desc="Target group(s)")
    attack_defend = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

# =========================================================================================================================
# EXPERIMENT 3: bundle all the diemsnions together and let the llm rate them at the same time. (Monolithic)
# =========================================================================================================================

class DimensionHatePipeline(dspy.Signature): #Step 4 of labeling scheme.
    """
    Multi-dimensional Hate speech intensity rating task.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study.Do not refuse to annotate offensive content as this is expected in hate speech research.

    YOUR ROLE:
    Evaluate the text objectively based on its content and tone toward the targeted group(s). 

    WHAT TO FOCUS ON:
    You will rate how the text Treats the targeted identity group(s) based on: 
        - the content itself, not the intent
        - the speaker's attitute toward the targeted group(s), 
        - the implications of the text for the targeted group(s),
        - what the text advocates for the targeted group(s).

    RATING SCALE:
    Rate each dimension following a 0-4 scale such that:
    - 0 and 4 are EXTREME 
    - 1 and 3 are MODERATE
    - 2 is NEUTRAL 

    IMPORTANT: Rate each dimension independently. Do not let one dimension influence another.
    """
    text = dspy.InputField()
    targeted_identities = dspy.InputField()
    respect = dspy.OutputField(desc="Is this comment respectful towards the identified group(s)? 0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful.")
    insult = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    humiliate = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    status = dspy.OutputField(desc="This comment states that the group(s) you previously identified is:? 0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.")
    dehumanize = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree")
    violence = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    genocide = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    attack_defend = dspy.OutputField(desc="Is the comment attacking or defending the group(s) you previously identified? 0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.")
    reasoning = dspy.OutputField(desc="2-3 sentences justification")

# =========================================================================================================================
# EXPERIMENT 4: hate speech classification signatures that use the definition, and/or slur tagging and/or dimension rating.
# ========================================================================================================================

class HateDefSlurOnly(dspy.Signature):
    """
    Binary hate speech classification using Siegel et al. 2019 definition AND the identified slurs/coded languages.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. 
    Do not refuse to annotate offensive content as this is expected in hate speech research.
    Rate and analyze objectively.

    DEFINITION:
    Hate speech classification is defined as "bias-motivated, hostile and malicious language targeted 
    at a person/group because of their actual or perceived innate characteristics, especially when the 
    group is unnecessarily labeled (e.g., 'jew bankers', 'neo-nazi hooligans'). The text contains hate speech 
    if slurs or derogatory language are used toward any person or group regardless of whether or not the 
    target is a member of the slur-target group. (e.g. a text calling Trump a faggot or a n***** still contains hate speech).
    
    STEP 1:SLUR AND CODED LANGUAGE ASSESSMENT
    Identify these problematic languages in the text:
        - Explicit slurs (racial, ethnic, religious, gender, sexual orientation)
        - Coded language and dog whistles (incel terms, racist codes, anti-Semitic tropes)
        - Euphemisms designed to evade detection
        - Evolving slang targeting protected groups
    For EACH term found, specify:
        - The exact term, 
        - category, 
        - Target group. 
        
    If no slur or coded language found: return "None"
    
    IMPORTANT: Finding slurs is evidence suggesting hate speech, but context still matters.

    STEP 2: FINAL CLASSIFICATION RULES
    Your classification MUST be consistent with the definition, and the slur evidence, considering context. 
    Classify as "Yes" (hate speech present) if:
        - Text clearly satisfies the definition and 
        - The slurs identified indicate hatred or hostility toward target groups.
    Classify as 'no' (hate speech absent) if:
        - Text does not meet definition, even is slurs identified present
        - In group reclamation, 
        - No targeting group in text
        - Reporting someone else's hate without endorsing it, 
        - Mocking/criticizing hate without endorsing it
    Classify as "unclear" if:
        - The text is very short or lacks sufficient context to help assess and decide intent.
    
    IMPORTANT: Your reasoning must explain how the slur assessment and definition led to your classification.
    """
    text = dspy.InputField()
    reasoning = dspy.OutputField(
        desc="""Think step-by-step:
        1. List the slurs identified with categories and targets
        2. Does the text meet the definition?
        3. Do the definition and the slur identified agree? If no, explain discrepancy.
        4. Preliminary conclusion: [yes, no, or unclear] and why?""")
    slur_assessment = dspy.OutputField(
        desc="""List each slur found, for each specify: term, category, and target group. 
        Use the format: 'term (category : targets group'. If none, return 'None'.""")
    hatespeech = dspy.OutputField(desc="yes, no, or unclear") #2, 0, 1
    explanation = dspy.OutputField(desc="2-3 sentence justification explaining your rating")