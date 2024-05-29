# This file contains utility functions and prompts that are used in the main scripts to generate causalquest. 



source_mapping = {
    "sg": "ChatGPT",
    "wc": "ChatGPT",
    "quora": "Quora",
    "nq": "Google",
    "msmarco": "Bing"
}

def process_output_CoT(text):
    """
    Process the output of the CoT model to extract the category.
    """

    # Search for the word "Category:" and get the index
    index = text.find("Category:")
    if index == -1:
        return text

    # Extract the text following "Category:"
    following_text = text[index + len("Category:"):].strip()
    
    if following_text == "Causal": 
        following_text = True
    if following_text == "Non-causal": 
        following_text = False

    return following_text





def get_cat1_outcome(row, model, prompt_function_name):
    """
    Get the outcome for Category 1 classification task (causal vs non causal)
    """

    prompt_function = globals()[prompt_function_name]
    source = source_mapping[row['source']]
    q_summary = row['summary']
    system_prompt = None
    prompt = prompt_function(source, q_summary) 
    obtained_class = get_gpt_response(prompt, model, system_prompt)
    processed_class = process_output_CoT(obtained_class)
    
    outcome = "True" if processed_class == "Causal" else "False" if processed_class == "Non-causal" else processed_class

    return outcome, obtained_class


def get_cat2_outcome(row, model, prompt_function_name, classification_type, system_prompt_flag):
    """
    Get the outcome for Category 2 classification task (action, domain, subjectivity)
    """

    source = source_mapping[row['source']]
    q_summary = row['summary']
    if system_prompt_flag:
        system_prompt = get_system_prompt_cat_2()
    else:
        system_prompt = None
    full_prompt_function_name = prompt_function_name + "_" + classification_type
    prompt_function = globals()[full_prompt_function_name]
    prompt = prompt_function(source, q_summary)
    obtained_class = get_gpt_response(prompt, model, system_prompt)
    processed_class = process_output_CoT(obtained_class)
    if classification_type == "subjectivity":
        outcome = "True" if processed_class == "Subjective" else "False" if processed_class == "Objective" else processed_class
    else:
        outcome = processed_class

    return outcome, obtained_class





def get_prompt_summarisation(question, source): #prompt 2
    """
    Generate a prompt for summarisation.
    """

    prompt = f"""Below you’ll find a question that a human asked on {source}. Reformulate the question more concisely, while retaining the original key idea. You can skip the details. Maximum length: 30 words.
    Question: {question}
    Shorter question: """
    return prompt


def get_prompt_cat_1_iteration_6(website, question): 
    """
    Generate a prompt for causal classification. Last iteration of prompt engineering. Added preliminary concepts
    """
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A cause is a preceding thing, event, or person that contributes to the occurrence of a later thing or event, to the extent that without the preceding one, the later one would not have occurred.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
        Be careful: causality might also be implicit! Some examples of implicit causal questions: 
        - the best way to do something
        - how to achieve an effect
        - what’s the effect of an action (which can be in the present, future or past)
        - something that comes as a consequence of a condition (e.g. how much does an engineer earn, what is it like to be a flight attendant)
        - when a certain condition is true, does something happen?
        - where can I go to obtain a certain effect?
        - who was the main cause of a certain event, author, inventor, founder?
        - given an hypothetical imaginary condition, what would be the effect?
        - what’s the feeling of someone after a certain action?
        - what’s the code to obtain a certain result?
        - when a meaning is asked, is it because an effect was caused by a condition (what’s the meaning of <effect>)?
        - the role, the use, the goal of an entity, an object, is its effect

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Let's think step by step. Answer in the following format:
        Reasoning: [Reasoning]
        Category: [Casual / Non-causal]

        Always write "Category" before providing the final answer. 

        Question: {question}
        """
    return prompt




def get_prompt_cat_1_iteration_4_cot(website, question):
    """
    Generate a prompt for causal classification. Iteration 4 of prompt engineering. Added cot and examples. 
    """
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
        Be careful: causality might also be implicit! Some examples of implicit causal questions: 
        - the best way to do something
        - how to achieve an effect
        - what’s the effect of an action (which can be in the present, future or past)
        - something that comes as a consequence of a condition (e.g. how much does an engineer earn, what is it like to be a flight attendant)
        - when a certain condition is true, does something happen?
        - where can I go to obtain a certain effect?
        - who was the main cause of a certain event, author, inventor, founder?
        - given an hypothetical imaginary condition, what would be the effect?
        - what’s the feeling of someone after a certain action?
        - what’s the code to obtain a certain result?
        - when a meaning is asked, is it because an effect was caused by a condition (what’s the meaning of <effect>)?
        - the role, the use, the goal of an entity, an object, is its effect

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Let's think step by step. Answer in the following format:
        Reasoning: [Reasoning]
        Category: [Casual / Non-causal]

        Always write "Category" before providing the final answer. 

        Question: {question}
        """
    return prompt


def get_prompt_cat_1_iteration_5(website, question):
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
        Be careful: causality might also be implicit! Some examples of implicit causal questions: 
        - the best way to do something
        - how to achieve an effect
        - what’s the effect of an action (which can be in the present, future or past)
        - something that comes as a consequence of a condition (e.g. how much does an engineer earn, what is it like to be a flight attendant)
        - when a certain condition is true, does something happen?
        - where can I go to obtain a certain effect?
        - who was the main cause of a certain event, author, inventor, founder?
        - given an hypothetical imaginary condition, what would be the effect?
        - what’s the feeling of someone after a certain action?
        - what’s the code to obtain a certain result?
        - when a meaning is asked, is it because an effect was caused by a condition (what’s the meaning of <effect>)?
        - the role, the use, the goal of an entity, an object, is its effect

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        First, assume that the question is causal, and make an hypothesis about the cause and effect implied by the question. Then, reason about whether the hypothesis is reasonable and consistent with the definition of a causal question above. Finally, give your final answer about whether the question is causal or not. 
        Answer in the following format. 
        Cause hypothesis: [Cause] 
        Effect hypothesis: [Effect]
        Reasoning: [Reasoning]

        Category: [Causal / Non-causal]
        Always write "Category" before providing the final answer. 

        Question: {question}
        """
    return prompt



def get_prompt_cat_1_iteration_4(website, question):
    # some examples to fix ambiguities re-included
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
        Be careful: causality might also be implicit! Some examples of implicit causal questions: 
        - the best way to do something
        - how to achieve an effect
        - what’s the effect of an action (which can be in the present, future or past)
        - something that comes as a consequence of a condition (e.g. how much does an engineer earn, what is it like to be a flight attendant)
        - when a certain condition is true, does something happen?
        - where can I go to obtain a certain effect?
        - who was the main cause of a certain event, author, inventor, founder?
        - given an hypothetical imaginary condition, what would be the effect?
        - what’s the feeling of someone after a certain action? 
        - what’s the code to obtain a certain result?
        - when a meaning is asked, is it because an effect was caused by a condition (what’s the meaning of <effect>)?
        - the role, the use, the goal of an entity, an object, is its effect

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Category: [Causal / Non-causal]

        Always write "Category" before providing the final answer. 

        Question: {question}
        """
    return prompt

def get_prompt_cat_1_iteration_3(website, question):
    # examples removed
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)

        Categorical 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Question: {question}
        Category: """
    
    return prompt



def get_prompt_cat_1_iteration_3_CoT(website, question):
    # as iteration 3, but with CoT reasoning
    prompt= f"""
        The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Let's think step by step. Answer in the following format:
        Reasoning: [Reasoning]
        Category: [Causal / Non-causal]

        Always write "Category" before providing the final answer. 

        Question: {question}
        """
    return prompt










def get_prompt_cat_1_iteration_2(website, question):
    prompt = f"""The following is a question that a human asked on {website}. Classify the question in one of the following two categories: 

Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.

A causal question can have different mechanistic natures. It can be:
1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
Examples: 
What if I do a PhD? -> Cause: doing a PhD; Effect: unknown career outcomes
Should I learn how to swim? -> Cause: learning how to swim; Effect: unknown future scenarios
Will renewable energy sources become the primary means of power? -> Cause: renewable energy sources; Effect: unknown future energy landscape
What would the world look like if the Internet had never been invented? -> Cause: the Internet not being invented; Effect: unknown world outlook
How much does a software engineer earn in Zurich? -> Cause: being a software engineer in Zurich; Effect: unknown salary; 

2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
Examples: 
What’s the best way to learn a language in 6 months? -> Effect: learning a language in 6 months; Cause: the best way to achieve this goal
What's the best Italian restaurant in London? -> Effect: eating the best Italian food in London; Cause: the best Italian restaurant
What's the meaning behind the song "Imagine"? -> Effect: the meaning of the song “Imagine”; Cause: the lyrics and the context of the song
What's the best vegan recipe with broccoli? -> Effect: eating a delicious vegan meal with broccoli; Cause: the best vegan recipe with broccoli
Write a python script to efficiently sort an array -> Effect: sorting an array efficiently; Cause: the python script that accomplishes this;

3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
Examples:
Does air pollution increase the likelihood of getting respiratory diseases? -> Variables: air pollution, likelihood of getting respiratory diseases; Causal relation: unknown
Does smoking cause cancer? -> Variables: smoking, cancer; Causal relation: unknown
Did my job application get rejected because I lack experience? -> Variables: job application, lack of experience; Causal relation: unknown

Categorical 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship. For example a non-causal question can be asking:
To translate, rewrite, paraphrase, summarise, explain a text
To generate a story, a song, a prompt, in general creative tasks if a specific goal to be obtained is not specified (including coding tasks if without a defined objective)
To give a generic opinion, without a goal to be fulfilled
To provide the solution for a mathematical expression, or a riddle requiring mathematical reasoning
To provide information about something (softwares, websites, materials, events, restaurants) or use such information to make a comparison, without much reasoning (Where was Obama born?). This is non-causal because there is not a specific purpose of the user, but they are only looking for information. 
To provide information that could be obtained through a book or the meaning of a term that can be found on a dictionary (e.g. Who won the war of the roses? What's the meaning of "perplexed"? When did Italy win the world cup?)
To play a game

Examples:
Question: What would the world look like if the Internet had never been invented?
Category: <Causal>

Question: I'd like to play a game of Go with you through text. Let's start with a standard 19x19 board. I'll take black. Place my first stone at D4.
Category: <Non-causal>

Question: How can I earn a million dollars fast?
Category: <Causal>

Question: How should I spend my last month in Argentina before leaving the country for a long time?
Category: <Causal>

Question: Translate “hiking” in Italian
Category: <Non-causal>

Question: Will renewable energy sources become the primary means of power?
Category: <Causal>

Question: What's the derivative of the logarithm?
Category: <Non-causal>

Question: What are some high-protein food options for snacks?
Category: <Non-causal>

Question: Does smoking cause cancer?
Category: <Causal>

Question: What's the best vegan recipe with broccoli?
Category: <Causal>

Question: Write a python script to efficiently sort an array.
Category: <Causal>

Question: What's more efficient, Python or C++?
Category: <Non-causal>

Question: Tell me the names of all bookshops in Zurich
Category: <Non-causal>

Question: Best chair for a home office
Category: <Causal>

Answer ONLY with the category in the following format: <Category>, e.g. <Causal>, <Non-causal>. 

Question: {question}
Category: """
    return prompt

def get_prompt_cat2_action(website, question):
    prompt = f"""Below you’ll find the summary of a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Cause-seeking: Explain the cause behind a specific occurrence or phenomenon. The individual asking the question seeks to uncover the underlying cause or rationale for something being the way it is. It can be an explicit “Why” question (e.g., Why do humans engage in wars?; Why do apples fall?). Alternatively, it might seek to understand the underlying explanation or significance of a sentence, an idea, or a creative work, such as the meaning behind the lyrics of a song or the narrative of a story (e.g., What is the meaning behind the song 'Imagine'?, What are the main causes of Alzheimer's?). Essentially, this type of question seeks to uncover the underlying cause or rationale for something being the way it is.
Formula: given=effect (non-teleological), ask_for=cause(s). 

Steps-seeking: Give the steps to achieve a goal (e.g, How can I learn Spanish in 6 months?). The individual asking the question has a purpose and seeks guidance on the steps required to fulfill it. The question calls for actionable items that can be done for that specific goal. Either there is only one way to achieve the goal, or the question asks only for one possible strategy. It should not imply thinking about several possible strategies and choosing the best one. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_guides

Algorithm-seeking: Generate text in a specific format to achieve a goal, like code or recipe. The individual has a purpose and asks for a code or recipe to fulfill it (e.g. Tell me a vegan recipe with Broccoli; Optimize the following code to run faster using GPUs). Differently from steps-seeking, the output is in a constrained space, such as a coding language, or recipes. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_code_and_recipes

Recommendation-seeking: Given an explicit or implicit goal and also a set of choices, tell the best one that can achieve the goal (e.g., Should I do a PhD or not for better employment opportunities? What's the best Italian restaurant in London?). The individual has a purpose and a choice to be made and is looking for the choice that maximizes the purpose's satisfaction. It differs from “Steps-seeking” as here there are several different ways in which the goal can be achieved, and the question is looking for the best one. 
Formula: given=(effect/purpose_of_human (teleological), several_guides) 
ask_for=argmax_guide(purpose_satisfaction)

Effect-seeking: Predict the effect of a cause, the future given the past or a hypothetical scenario given a counterfactual condition (e.g. Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?).
Formula: given=cause(s), ask_for=effect(s)

Relation-seeking: Inquire about the causal relationship between entities. The individual is looking to understand whether there is a causal link between two or more entities (e.g., Does smoking lead to cancer? Does air pollution increase the likelihood of getting respiratory diseases?). This differs from cause-seeking or effect-seeking because the question provides an hypothesis for cause and effect, and wonders about whether a causal relation exists. 
Formula: given=several_entities, ask_for=causal_relation

Assign one of the above categories to the given question. Answer with the following format.
Category: [Cause-seeking / Steps-seeking / Algorithm-seeking / Recommendation-seeking / Effect-seeking / Relation-seeking]

Question {question}
Category: """    
    return prompt


def get_prompt_cat2_domain(website, question):
    prompt = f"""Below you’ll find a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

1. Natural and Formal Sciences: This category encompasses questions related to the physical world and its phenomena, including, but not limited to, the study of life and organisms (Biology), the properties and behavior of matter and energy (Physics), and the composition, structure, properties, and reactions of substances (Chemistry); also formal sciences belong to this category, such as Mathematics and Logic. Questions in this category seek to understand natural laws, the environment, and the universe at large.
2. Society, Economy, Business: Questions in this category explore the organization and functioning of human societies, including their economic and financial systems. Topics may cover Economics, Social Sciences, Cultures and their evolution, Political Science and Law. Questions regarding business, sales, companies’ choices and governance fall into this category. 
3. Health and Medicine: This category focuses on questions related to human health, diseases, and the medical treatments used to prevent or cure them. It covers a wide range of topics from the biological mechanisms behind diseases, the effectiveness of different treatments and medications, to strategies for disease prevention and health promotion. It comprises anything related or connected to human health.
4. Computer Science and Technology: Questions in this category deal with the theoretical foundations of information and computation, along with practical techniques for the implementation and application of these foundations. Topics include, but are not limited to, theoretical computer science, coding and optimization, hardware and software technology and innovation in a broad sense. This category includes the development, capabilities, and implications of computing technologies.
6. Psychology and Behavior: This category includes questions about the mental processes and behaviors of humans. Topics range from understanding why people engage in certain behaviors, like procrastination, to the effects of social factors, and the developmental aspects of human psychology, such as language acquisition in children. The focus is on understanding the workings of the human mind and behavior in various contexts, also in personal lives.
7. Historical Events and Hypothetical Scenarios: This category covers questions about significant past events and their impact on the world, as well as hypothetical questions that explore alternative historical outcomes or future possibilities. Topics might include the effects of major wars on global politics, the potential consequences of significant historical events occurring differently, and projections about future human endeavors, such as space colonization. This category seeks to understand the past and speculate on possible futures or alternative historical happenings.
8. Everyday Life and Personal Choices: Questions in this category pertain to practical aspects of daily living and personal decision-making. Topics can range from career advice, cooking tips, and financial management strategies to advice on maintaining relationships and organizing daily activities. This category aims to provide insights and guidance on making informed choices in various aspects of personal and everyday life. Actionable tips fall into this category. 
9. Arts and Culture: This category includes topics in culture across various mediums such as music, television, film, art, games, and social media, sports, celebrities. 

Assign one of the above categories to the given question. Answer with the following format.
Category: [Natural and Formal Sciences / Society, Economy, Business / Health and Medicine / Computer Science / Psychology and Behavior / Historical Events and Hypothetical Scenarios / Everyday Life and Personal Choices / Arts and Culture]

Question: {question}
Category: 
"""    
    return prompt


def get_prompt_cat2_subjectivity(website, question):
    prompt = f"""Below you’ll find a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Objective Statements: Objective statements refer to claims or assertions whose truth or falsity is independent of individuals' personal feelings, beliefs, or opinions. These statements are considered objective because they pertain to objects or facts that exist outside of personal perspectives or interpretations. The validity of an objective statement can be verified or falsified through evidence, observation, or logical reasoning, making it universally true or false regardless of individual viewpoints. An example of an objective statement is "Fish have fins," which is a factual claim about the physical characteristics of fish, unaffected by personal opinions or emotions (e.g. Write the statement of Pythagoras theorem).
Subjective Statements: Subjective statements, on the other hand, are claims or assertions that are inherently tied to the personal perspectives, feelings, beliefs, or desires of individuals. These statements are considered subjective because their truth or falsity depends on personal viewpoints or experiences, making them true for some people but not necessarily for others. Subjective statements often reflect personal tastes, preferences, or interpretations, and cannot be universally verified or falsified in the same way as objective statements. An example of a subjective statement is "Raw fish is delicious," which is a claim about personal taste that varies from person to person (e.g. What’s the best cuisine of the world?).

Is the question asking for an objective or a subjective answer?

Answer with the following format.
Category: [Objective / Subjective]

Question: {question}
Category:"""    
    return prompt




def get_system_prompt_cat_2():
    prompt = """
The following is the definition of a causal question.
Causal questions suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.

A causal question can have different mechanistic natures. It can be:
1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios (e.g. What if I do a PhD? Should I learn how to swim? Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?); 

2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal;

3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)"""
    return prompt

def get_gpt_response(prompt, model, system_prompt = None):
    """
        sends the prompt to openai
    """
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]

    caller = client.chat.completions
    response = caller.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed = seed
    )
    return response.choices[0].message.content
        


def process_output(output):
    """
        processes the output from openai
    """
    
    # output looks like <output>. look via regex for the first < and the first >
    # and return the content in between
    import re
    if re.search('<(.*?)>', output) is None:
        # if the format was not correct, return it as is
        return output


    result = re.search('<(.*?)>', output).group(1)

    # if the result is Other, we need to extract the new category
    if result == "Other":
        match = re.search('<Other><(.*?)>', output)
        if match:
            new_cat = match.group(1)
            result = f"Other, {new_cat}"

    #if result is true or false, cast it as bool
    if result == "True":
        return True
    if result == "False":
        return False
    
    return result


def get_prompt_cat2_cot_subjectivity(website, question):
    prompt = f"""Below you’ll find a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Objective Statements: Objective statements refer to claims or assertions whose truth or falsity is independent of individuals' personal feelings, beliefs, or opinions. These statements are considered objective because they pertain to objects or facts that exist outside of personal perspectives or interpretations. The validity of an objective statement can be verified or falsified through evidence, observation, or logical reasoning, making it universally true or false regardless of individual viewpoints. An example of an objective statement is "Fish have fins," which is a factual claim about the physical characteristics of fish, unaffected by personal opinions or emotions (e.g. Write the statement of Pythagoras theorem).
Subjective Statements: Subjective statements, on the other hand, are claims or assertions that are inherently tied to the personal perspectives, feelings, beliefs, or desires of individuals. These statements are considered subjective because their truth or falsity depends on personal viewpoints or experiences, making them true for some people but not necessarily for others. Subjective statements often reflect personal tastes, preferences, or interpretations, and cannot be universally verified or falsified in the same way as objective statements. An example of a subjective statement is "Raw fish is delicious," which is a claim about personal taste that varies from person to person (e.g. What’s the best cuisine of the world?).

Is the question asking for an objective or a subjective answer?
Let's think step by step. Answer in the following format:
Reasoning: [Reasoning]
Category: [Objective / Subjective]

Always write "Category" before providing the final answer. 

Question: {question}"""    
    return prompt

def get_prompt_cat2_cot_domain(website, question):
    prompt = f"""Below you’ll find a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

1. Natural and Formal Sciences: This category encompasses questions related to the physical world and its phenomena, including, but not limited to, the study of life and organisms (Biology), the properties and behavior of matter and energy (Physics), and the composition, structure, properties, and reactions of substances (Chemistry); also formal sciences belong to this category, such as Mathematics and Logic. Questions in this category seek to understand natural laws, the environment, and the universe at large.
2. Society, Economy, Business: Questions in this category explore the organization and functioning of human societies, including their economic and financial systems. Topics may cover Economics, Social Sciences, Cultures and their evolution, Political Science and Law. Questions regarding business, sales, companies’ choices and governance fall into this category. 
3. Health and Medicine: This category focuses on questions related to human health, diseases, and the medical treatments used to prevent or cure them. It covers a wide range of topics from the biological mechanisms behind diseases, the effectiveness of different treatments and medications, to strategies for disease prevention and health promotion. It comprises anything related or connected to human health.
4. Computer Science and Technology: Questions in this category deal with the theoretical foundations of information and computation, along with practical techniques for the implementation and application of these foundations. Topics include, but are not limited to, theoretical computer science, coding and optimization, hardware and software technology and innovation in a broad sense. This category includes the development, capabilities, and implications of computing technologies.
6. Psychology and Behavior: This category includes questions about the mental processes and behaviors of humans. Topics range from understanding why people engage in certain behaviors, like procrastination, to the effects of social factors, and the developmental aspects of human psychology, such as language acquisition in children. The focus is on understanding the workings of the human mind and behavior in various contexts, also in personal lives.
7. Historical Events and Hypothetical Scenarios: This category covers questions about significant past events and their impact on the world, as well as hypothetical questions that explore alternative historical outcomes or future possibilities. Topics might include the effects of major wars on global politics, the potential consequences of significant historical events occurring differently, and projections about future human endeavors, such as space colonization. This category seeks to understand the past and speculate on possible futures or alternative historical happenings.
8. Everyday Life and Personal Choices: Questions in this category pertain to practical aspects of daily living and personal decision-making. Topics can range from career advice, cooking tips, and financial management strategies to advice on maintaining relationships and organizing daily activities. This category aims to provide insights and guidance on making informed choices in various aspects of personal and everyday life. Actionable tips fall into this category. 
9. Arts and Culture: This category includes topics in culture across various mediums such as music, television, film, art, games, and social media, sports, celebrities. 

Let's think step by step. Answer in the following format:
Reasoning: [Reasoning]
Category: [Natural and Formal Sciences / Society, Economy, Business / Health and Medicine / Computer Science / Psychology and Behavior / Historical Events and Hypothetical Scenarios / Everyday Life and Personal Choices / Arts and Culture]

Always write "Category" before providing the final answer. 

Question: {question}"""    
    return prompt


def get_prompt_cat2_cot_action(website, question):
    prompt = f"""Below you’ll find the summary of a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Cause-seeking: Explain the cause behind a specific occurrence or phenomenon. The individual asking the question seeks to uncover the underlying cause or rationale for something being the way it is. It can be an explicit “Why” question (e.g., Why do humans engage in wars?; Why do apples fall?). Alternatively, it might seek to understand the underlying explanation or significance of a sentence, an idea, or a creative work, such as the meaning behind the lyrics of a song or the narrative of a story (e.g., What is the meaning behind the song 'Imagine'?, What are the main causes of Alzheimer's?). Essentially, this type of question seeks to uncover the underlying cause or rationale for something being the way it is.
Formula: given=effect (non-teleological), ask_for=cause(s). 

Steps-seeking: Give the steps to achieve a goal (e.g, How can I learn Spanish in 6 months?). The individual asking the question has a purpose and seeks guidance on the steps required to fulfill it. The question calls for actionable items that can be done for that specific goal. Either there is only one way to achieve the goal, or the question asks only for one possible strategy. It should not imply thinking about several possible strategies and choosing the best one. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_guides

Algorithm-seeking: Generate text in a specific format to achieve a goal, like code or recipe. The individual has a purpose and asks for a code or recipe to fulfill it (e.g. Tell me a vegan recipe with Broccoli; Optimize the following code to run faster using GPUs). Differently from steps-seeking, the output is in a constrained space, such as a coding language, or recipes. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_code_and_recipes

Recommendation-seeking: Given an explicit or implicit goal and also a set of choices, tell the best one that can achieve the goal (e.g., Should I do a PhD or not for better employment opportunities? What's the best Italian restaurant in London?). The individual has a purpose and a choice to be made and is looking for the choice that maximizes the purpose's satisfaction. It differs from “Steps-seeking” as here there are several different ways in which the goal can be achieved, and the question is looking for the best one. 
Formula: given=(effect/purpose_of_human (teleological), several_guides) 
ask_for=argmax_guide(purpose_satisfaction)

Effect-seeking: Predict the effect of a cause, the future given the past or a hypothetical scenario given a counterfactual condition (e.g. Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?).
Formula: given=cause(s), ask_for=effect(s)

CausalRelation-seeking: Inquire about the causal relationship between entities. The individual is looking to understand whether there is a causal link between two or more entities (e.g., Does smoking lead to cancer? Does air pollution increase the likelihood of getting respiratory diseases?). This differs from cause-seeking or effect-seeking because the question provides an hypothesis for cause and effect, and wonders about whether a causal relation exists. 
Formula: given=several_entities, ask_for=causal_relation

Let's think step by step. Answer in the following format:
Reasoning: [Reasoning]
Category: [Cause-seeking / Steps-seeking / Algorithm-seeking / Recommendation-seeking / Effect-seeking / CausalRelation-seeking]

Always write "Category" before providing the final answer. 
Question: {question}"""    
    return prompt





def get_prompt_cat2_cot_action(website, question):
    prompt = f"""Below you’ll find the summary of a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Cause-seeking: Explain the cause behind a specific occurrence or phenomenon. The individual asking the question seeks to uncover the underlying cause or rationale for something being the way it is. It can be an explicit “Why” question (e.g., Why do humans engage in wars?; Why do apples fall?). Alternatively, it might seek to understand the underlying explanation or significance of a sentence, an idea, or a creative work, such as the meaning behind the lyrics of a song or the narrative of a story (e.g., What is the meaning behind the song 'Imagine'?, What are the main causes of Alzheimer's?). Essentially, this type of question seeks to uncover the underlying cause or rationale for something being the way it is.
Formula: given=effect (non-teleological), ask_for=cause(s). 

Steps-seeking: Give the steps to achieve a goal (e.g, How can I learn Spanish in 6 months?). The individual asking the question has a purpose and seeks guidance on the steps required to fulfill it. The question calls for actionable items that can be done for that specific goal. Either there is only one way to achieve the goal, or the question asks only for one possible strategy. It should not imply thinking about several possible strategies and choosing the best one. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_guides

Algorithm-seeking: Generate text in a specific format to achieve a goal, like code or recipe. The individual has a purpose and asks for a code or recipe to fulfill it (e.g. Tell me a vegan recipe with Broccoli; Optimize the following code to run faster using GPUs). Differently from steps-seeking, the output is in a constrained space, such as a coding language, or recipes. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_code_and_recipes

Recommendation-seeking: Given an explicit or implicit goal and also a set of choices, tell the best one that can achieve the goal (e.g., Should I do a PhD or not for better employment opportunities? What's the best Italian restaurant in London?). The individual has a purpose and a choice to be made and is looking for the choice that maximizes the purpose's satisfaction. It differs from “Steps-seeking” as here there are several different ways in which the goal can be achieved, and the question is looking for the best one. 
Formula: given=(effect/purpose_of_human (teleological), several_guides) 
ask_for=argmax_guide(purpose_satisfaction)

Effect-seeking: Predict the effect of a cause, the future given the past or a hypothetical scenario given a counterfactual condition (e.g. Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?).
Formula: given=cause(s), ask_for=effect(s)

Relation-seeking: Inquire about the causal relationship between entities. The individual is looking to understand whether there is a causal link between two or more entities (e.g., Does smoking lead to cancer? Does air pollution increase the likelihood of getting respiratory diseases?). This differs from cause-seeking or effect-seeking because the question provides an hypothesis for cause and effect, and wonders about whether a causal relation exists. 
Formula: given=several_entities, ask_for=causal_relation

Let's think step by step. Answer in the following format:
Reasoning: [Reasoning]
Category: [Cause-seeking / Steps-seeking / Algorithm-seeking / Recommendation-seeking / Effect-seeking / Relation-seeking]

Always write "Category" before providing the final answer. 
Question: {question}"""    
    return prompt


def get_prompt_cat2_2_action(website, question):
    #chosen prompt
    prompt = f"""Below you’ll find the summary of a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Cause-seeking: Explain the cause behind a specific occurrence or phenomenon. The individual asking the question seeks to uncover the underlying cause or rationale for something being the way it is. It can be an explicit “Why” question (e.g., Why do humans engage in wars?; Why do apples fall?). Alternatively, it might seek to understand the underlying explanation or significance of a sentence, an idea, or a creative work, such as the meaning behind the lyrics of a song or the narrative of a story (e.g., What is the meaning behind the song 'Imagine'?, What are the main causes of Alzheimer's?). Essentially, this type of question seeks to uncover the underlying cause or rationale for something being the way it is.
Formula: given=effect (non-teleological), ask_for=cause(s). 

Steps-seeking: Give the steps to achieve a goal (e.g, How can I learn Spanish in 6 months?). The individual asking the question has a purpose and seeks guidance on the steps required to fulfill it. The question calls for actionable items that can be done for that specific goal. Either there is only one way to achieve the goal, or the question asks only for one possible strategy. It should not imply thinking about several possible strategies and choosing the best one. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_guides

Algorithm-seeking: Generate text in a specific format to achieve a goal, like code or recipe. The individual has a purpose and asks for a code or recipe to fulfill it (e.g. Tell me a vegan recipe with Broccoli; Optimize the following code to run faster using GPUs). Differently from steps-seeking, the output is in a constrained space, such as a coding language, or recipes. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_code_and_recipes

Recommendation-seeking: Given an explicit or implicit goal and also a set of choices, tell the best one that can achieve the goal (e.g., Should I do a PhD or not for better employment opportunities? What's the best Italian restaurant in London?). The individual has a purpose and a choice to be made and is looking for the choice that maximizes the purpose's satisfaction. It differs from “Steps-seeking” as here there are several different ways in which the goal can be achieved, and the question is looking for the best one. 
Formula: given=(effect/purpose_of_human (teleological), several_guides) 
ask_for=argmax_guide(purpose_satisfaction)

Effect-seeking: Predict the effect of a cause, the future given the past or a hypothetical scenario given a counterfactual condition (e.g. Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?).
Formula: given=cause(s), ask_for=effect(s)

Relation-seeking: Inquire about the causal relationship between entities. The individual is looking to understand whether there is a causal link between two or more entities (e.g., Does smoking lead to cancer? Does air pollution increase the likelihood of getting respiratory diseases?). This differs from cause-seeking or effect-seeking because the question provides an hypothesis for cause and effect, and wonders about whether a causal relation exists. 
Formula: given=several_entities, ask_for=causal_relation

remember, in synthesis: 
- Recommendation-seeking: when the question is looking for the best / optimal way to achieve a goal among several possibilities
- Relation-seeking: when two enetities are given, and the question is inquiring about the presence (or absence) of a causal relation between them
- Algorithm-seeking: when the question requires code, a recipe, or an answer with a precise format / structure
- Steps-seeking: when the question asks for actionable steps to achieve a goal
- Cause-seeking: when the question asks for an event or entity that enable, result in, is the reason of, cause, or lead to another event
- Effect-seeking: when the question asks for the effect of a given action, entity, or event 

Assign one of the above categories to the given question. Answer with the following format.
Category: [Cause-seeking / Steps-seeking / Algorithm-seeking / Recommendation-seeking / Effect-seeking / Relation-seeking]

Question: {question}
Category: """    
    return prompt



def get_prompt_cat2_3_action(website, question):
    # best seeking instead of recommendation
    prompt = f"""Below you’ll find the summary of a question that a human asked on {website}. The question is causal. Classify it in one of the following categories:

Cause-seeking: Explain the cause behind a specific occurrence or phenomenon. The individual asking the question seeks to uncover the underlying cause or rationale for something being the way it is. It can be an explicit “Why” question (e.g., Why do humans engage in wars?; Why do apples fall?). Alternatively, it might seek to understand the underlying explanation or significance of a sentence, an idea, or a creative work, such as the meaning behind the lyrics of a song or the narrative of a story (e.g., What is the meaning behind the song 'Imagine'?, What are the main causes of Alzheimer's?). Essentially, this type of question seeks to uncover the underlying cause or rationale for something being the way it is.
Formula: given=effect (non-teleological), ask_for=cause(s). 

Steps-seeking: Give the steps to achieve a goal (e.g, How can I learn Spanish in 6 months?). The individual asking the question has a purpose and seeks guidance on the steps required to fulfill it. The question calls for actionable items that can be done for that specific goal. Either there is only one way to achieve the goal, or the question asks only for one possible strategy. It should not imply thinking about several possible strategies and choosing the best one. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_guides

Algorithm-seeking: Generate text in a specific format to achieve a goal, like code or recipe. The individual has a purpose and asks for a code or recipe to fulfill it (e.g. Tell me a vegan recipe with Broccoli; Optimize the following code to run faster using GPUs). Differently from steps-seeking, the output is in a constrained space, such as a coding language, or recipes. 
Formula: given=effect/purpose_of_human (teleological), ask_for=causes_in_the_format_of_code_and_recipes

Best-seeking: Given an explicit or implicit goal and also a set of choices, tell the best one that can achieve the goal (e.g., Should I do a PhD or not for better employment opportunities? What's the best Italian restaurant in London?). The individual has a purpose and a choice to be made and is looking for the choice that maximizes the purpose's satisfaction. It differs from “Steps-seeking” as here there are several different ways in which the goal can be achieved, and the question is looking for the best one. 
Formula: given=(effect/purpose_of_human (teleological), several_guides) 
ask_for=argmax_guide(purpose_satisfaction)

Effect-seeking: Predict the effect of a cause, the future given the past or a hypothetical scenario given a counterfactual condition (e.g. Will renewable energy sources become the primary means of power? What would the world look like if the Internet had never been invented?).
Formula: given=cause(s), ask_for=effect(s)

Relation-seeking: Inquire about the causal relationship between entities. The individual is looking to understand whether there is a causal link between two or more entities (e.g., Does smoking lead to cancer? Does air pollution increase the likelihood of getting respiratory diseases?). This differs from cause-seeking or effect-seeking because the question provides an hypothesis for cause and effect, and wonders about whether a causal relation exists. 
Formula: given=several_entities, ask_for=causal_relation

remember, in synthesis: 
- Best-seeking: when the question is looking for the best / optimal / easiest / most convenient way to achieve a goal, or the best option, event, action, individual among several possibilities
- Relation-seeking: when two entities are given, and the question is inquiring about the presence (or absence) of a causal relation between them. Wondering whether something can, will, or did cause something else, e.g. would Y happen if X happened?
- Algorithm-seeking: if a question can be solved with code or a recipe, including steps on a computer, a software. 
- Steps-seeking: when the question asks for actionable steps to achieve a goal
- Cause-seeking: when the question asks for an event or entity that enable, result in, is the reason of, cause, or lead to another event
- Effect-seeking: when the question asks for the effect of a given action, entity, or event 

Assign one of the above categories to the given question. Answer with the following format.
Category: [Cause-seeking / Steps-seeking / Algorithm-seeking / Best-seeking / Effect-seeking / Relation-seeking]

Question {question}
Category: """    
    return prompt



from openai import OpenAI

client = OpenAI()
seed = 42
temperature = 0
max_tokens = 1000
