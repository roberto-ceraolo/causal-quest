# this file contains the functions for the linguistic baseline of CausalQuest

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
import jsonlines
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
import re

# Based on section 3 
causal_connective = {'because of', 'thanks to', 'due to', 'because'}
causal_verbs_sec3 = {'cause', 'lead to', 'bring about', 'generate', 'make', 'force', 'help', 'let', 'prevent',
                'allow', 'kill', 'melt', 'dry', 'break','drop', 
               'poison', 'hang', 'punch', 'clean'}

# Simple list of causative verbs as defined in English Grammar Books (esp for EAL), get, have, help, let, make, and prevent



# Based on section 4, procedure 1 of Giru and Moldovan
causal_lexico_syntactic = {'give rise to', 'induce', 'produce', 'generate', 'effect', 'bring about', 
            'provoke', 'arouse', 'elicit', 'lead to', 'trigger', 'derive from', 'associate with', 'relate to',
            'link to', 'stem from', 'originate', 'bring forth', 'lead up', 'trigger off', 'bring on', 'result', 'result from',
            'stir up', 'entail', 'contribute to', 'set up', 'trigger off', 'commence', 'set off', 'set in motion', 'bring on', 'conduce', 'educe',
            'originate in', 'lead off', 'spark', 'spark off', 'evoke', 'link up', 'implicate', 'activate', 'actuate',
            'kindle', 'fire up', 'stimulate', 'call forth', 'unleash', 'effectuate', 'kick up', 'give birth', 'call down',
            'put forward', 'create', 'launch', 'develop', 'bring', 'make', 'begin', 'rise'
           }

# Morphological causatives 

morph_causatives = {
    'blacken', 'brighten', 'broaden', 'deepen', 'darken', 'harden', 'heighten',
    'lengthen', 'lighten', 'loosen', 'redden', 'sharpen', 'shorten', 'strengthen',
    'sweeten', 'thicken', 'tighten', 'toughen', 'widen', 'brighten', 'amplify',
    'clarify', 'classify', 'deify', 'dignify', 'diversify', 'electrify', 'glorify',
    'horrify', 'identify', 'intensify', 'justify', 'magnify', 'mortify', 'nullify',
    'pacify', 'purify', 'qualify', 'simplify', 'solidify', 'terrify', 'verify',
    'liquefy', 'enlighten', 'flatten', 'lengthen', 'moisten', 'quicken'
}

causal_adverbs = {'audibly', 'visbly', 'manifestly', 'patently'
                 'publicly', 'conspicuously', 'successfully', 'plausibly', 'conveniently', 
                  'amusingly', 'pleasantly', 'irrevocably', 'tenuously', 'precariously', 'rudely',
                  'obediently', 'gratefully', 'consequently', 'painfully',
                  'mechanically', 'magically'}


causal_phrases = [
    "what causes", "what leads to", "what results in", "what brings about", 
]


# Regular expressions for each pattern as described in CausalQA, table 3 
patterns = {
    'R1': r'\bwhy\b',
    'R2': r'\bcause(s)?\b',
    'R3': r'\bhow (come|did)\b',
    'R4': r'\beffect(s)?\b|\baffect(s)?\b',
    'R5': r'\blead(s)? to\b',
    'R6': r'\bwhat (will|might)? happen(s)?\b(\bif\b|\bwhen\b)',
    'R7': r'\bwhat (to do|should be done)\b(\bif\b|\bto\b|\bwhen\b)'
}

def get_synonyms_antonyms(seed_set, degrees):
    """
    Get synonyms and antonyms of a given set of words. 
    The function will return a set of words that includes the seed set, synonyms and antonyms of the seed set.
    The degrees parameter specifies the depth of the search in the WordNet graph.
    """

    def find_related_words(word, degrees_left):
        """
        Find synonyms and antonyms of a given word.
        """
        if degrees_left == 0:
            return set()
        
        related_words = set()
        for synset in wn.synsets(word, pos=wn.VERB):
            # Finding synonyms
            for lemma in synset.lemmas():
                synonym = lemma.name()
                if synonym != word:
                    related_words.add(synonym)
                    related_words.update(find_related_words(synonym, degrees_left - 1))
                    
                # Finding antonyms
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonym = ant.name()
                        related_words.add(antonym)
                        related_words.update(find_related_words(antonym, degrees_left - 1))
                        
        return related_words
    
    final_synonyms_antonyms = set(seed_set)
    for word in seed_set:
        final_synonyms_antonyms.update(find_related_words(word, degrees))
    
    return final_synonyms_antonyms

def has_causal_structure(doc, causal_verbs):
    """
    Check if the given document has a causal structure.
    """
    for token in doc:
        if token.dep_ in {"nsubj", "ccomp"} and token.lemma_.lower() in causal_verbs:
            return True
    return False

def match_causal_patterns(doc):
    """
    Check if the given document matches any of the causal patterns.
    """
    for token in doc:
        if token.text.lower() == "what":
            next_token = token.nbor()
            if next_token.lemma_.lower() in {"cause", "lead", "result", "bring", "make", "induce", "trigger"}:
                return True
    return False





def match_patterns(sentence):
    """
    Check if the given sentence matches any of the causal patterns.
    """
    matches = []
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            matches.append(pattern_name)
    return matches

def is_causal_question(question, causal_verbs, causal_keywords):
    """
    Check if the given question is a causal question.
    """
    doc = nlp(question)

    if len(doc) < 2:
        return False
    
    for token in doc:
        if token.lemma_.lower() in causal_keywords:
            return True
        
    if doc[0].lemma_.lower() in {"why", "how"}:
        return True

    if has_causal_structure(doc, causal_verbs):
        return True
    

    if match_patterns(question):
        return True
    
    return False


def classify_and_compute(jsonl_input_file, jsonl_output_file):
    """
    Classify the given sentences as causal or non-causal and compute the accuracy and precision of the classification.
    
    """
    total_sentences = 0
    true_positives = 0
    false_positives = 0

    synonyms_antonyms = get_synonyms_antonyms(causal_verbs_sec3, 2)
    causal_verbs = causal_lexico_syntactic.union(morph_causatives,synonyms_antonyms)
    causal_keywords = causal_connective.union(causal_adverbs)
    
    with jsonlines.open(jsonl_input_file, 'r') as reader, jsonlines.open(jsonl_output_file, 'w') as writer:
        for obj in reader:
            sentence = obj.get('query')
            gold_label = obj.get('is_causal')
            
            if sentence and gold_label is not None:
                total_sentences += 1
                predicted_label = is_causal_question(sentence, causal_verbs, causal_keywords)
                obj['predicted_is_causal'] = predicted_label
                writer.write(obj)
                
                if predicted_label == gold_label:
                    if predicted_label:
                        true_positives += 1
                else:
                    if predicted_label:
                        false_positives += 1
    
    accuracy = true_positives / total_sentences if total_sentences > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    print(f"Total sentences: {total_sentences}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Classification results saved to: {jsonl_output_file}")


def inspect_false_positives(jsonl_output_file, false_positives_output_file, misclassified_output_file):
    """
    Inspect the false positives and misclassified instances in the output file.
    """

    false_positives = []
    misclassified = []

    with jsonlines.open(jsonl_output_file, 'r') as reader:
        for obj in reader:
            sentence = obj.get('query')
            gold_label = obj.get('is_causal')
            predicted_label = obj.get('predicted_is_causal')
            
            if gold_label and predicted_label:
                if predicted_label and not gold_label:
                    false_positives.append(obj)
                elif gold_label and not predicted_label:
                    misclassified.append(obj)
    
    # Write false positives to a separate output file
    with jsonlines.open(false_positives_output_file, 'w') as writer:
        for fp in false_positives:
            writer.write(fp)

    # Write misclassified instances to a separate output file
    with jsonlines.open(misclassified_output_file, 'w') as writer:
        for mc in misclassified:
            writer.write(mc)









# discovery of lexico syntactic patterns 
"""
Input: semantic relation R
Output: list of lexico-syntactic patterns expresssing R 

1. Pick the relation R = causation 
2. Pick a pair of noun phrases Ci, Cj among which R holds. 

For wordnet: is-a, reverse is-a, meronymy/holonym, entail, cause-to, attribute, pertainymy, antonymy, synset 

The cause-to is a transitive relation between verb synsets 

3. Extract lexico-syntactic patterns that link the two selected noun phrases 

"""

def get_causative_pairs():
    """
    Get causative pairs from WordNet.
    """

    causative_pairs = []
    
    for synset in wn.all_synsets('v'):  # 'v' for verbs
        for lemma in synset.lemmas():
            if lemma.derivationally_related_forms():
                for related_lemma in lemma.derivationally_related_forms():
                    if 'cause' in related_lemma.name():
                        causative_pairs.append((lemma.name(), related_lemma.name()))
    return causative_pairs

def extract_lexico_syntactic_patterns(causative_pairs):
    """
    Extract lexico-syntactic patterns that link the two selected noun phrases.
    """
    patterns = []
    
    for verb1, verb2 in causative_pairs:
        synset1 = wn.synsets(verb1, pos='v')[0]
        synset2 = wn.synsets(verb2, pos='v')[0]
        
        if synset1 and synset2:
            hypernyms1 = synset1.hypernyms()
            hypernyms2 = synset2.hypernyms()
            
            for hypernym1 in hypernyms1:
                for hypernym2 in hypernyms2:
                    if hypernym1 == hypernym2:
                        pattern = f"{verb1} causes {verb2}"
                        patterns.append(pattern)
                        break
    return patterns


def main_lexico_syntactic_patterns():
    """
    Main function to extract lexico-syntactic patterns.
    """
    # Step 1
    relation = 'cause-to'

    # Step 2: Pick a pair of noun phrases Ci, Cj among which R holds
    causative_pairs = get_causative_pairs()
    print(causative_pairs)

    # Step 3: Extract lexico-syntactic patterns that link the two selected noun phrases
    patterns = extract_lexico_syntactic_patterns(causative_pairs)

    # Output the patterns
    print("Lexico-Syntactic Patterns expressing causation:")
    for pattern in patterns:
        print(pattern)
