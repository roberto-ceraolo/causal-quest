# Scripts to reproduce the graphs and the linguistic analysis table of CausalQuest paper
import pandas as pd
import seaborn as sns
save = True
plots_folder = '../plots/'

import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def sunburst_iscausal(db):
    palette = sns.color_palette("Blues", 30)
    # Convert RGB colors to HEX
    hex_colors = ["#{:02x}{:02x}{:02x}".format(int(r*255),int(g*255),int(b*255)) for r,g,b in palette]

    evenly_spaced_indices = [int(i * (len(hex_colors) / 3)) for i in range(3)]
    selected_evenly_spaced_colors = [hex_colors[i] for i in evenly_spaced_indices]


    # Define the parent and child categories
    parent_categories = db.is_causal.value_counts().index.tolist()
    mapping = {True: 'Causal', False: 'Not Causal'}
    parent_categories = [mapping[category] for category in parent_categories]

    child_categories = db['action_class'].value_counts().index.tolist()

    # Calculate the sizes of the parent and child categories
    parent_sizes = db.is_causal.value_counts().values.tolist()
    child_sizes = db['action_class'].value_counts().values.tolist()

    # Combine the sizes of the parent and child categories
    size_data = parent_sizes + child_sizes

    # Create the data dictionary for the sunburst chart
    data = dict(
        categories=parent_categories + child_categories,
        parent=[''] * len(parent_categories) + ['Causal'] * len(child_categories),
        value=size_data
    )

    # Define the colors for the parent and child categories

    colors = [hex_colors[11]] + [hex_colors[24]] + [hex_colors[8]] * len(child_categories)


    # Create the sunburst chart
    fig = px.sunburst(
        data,
        names='categories',
        parents='parent',
        values='value',
        branchvalues='total',
        color='categories',
        color_discrete_map={category: color for category, color in zip(data['categories'], colors)}
    )

    # Update the layout of the sunburst chart
    fig.update_layout(autosize=False,
        margin = {'l':0,'r':0,'t':0,'b':0},
                    font=dict(size=25),
                    width=1200,  # Width of the figure in pixels
                    height=800   # Height of the figure in pixels
                    )

    fig.update_traces(textinfo="percent parent+label")


    # save in pdf
    if save: 
        fig.write_image(plots_folder + "action_class_sunburst.pdf", width=1200, height=800, scale=1)

    fig.show()


font_size = 50
def add_line_breaks(labels, max_length=20):
    new_labels = []
    for label in labels:
        if len(label) > max_length and label!="Psychology and Behavior":
            words = label.split()
            mid_point = len(words) // 2
            new_label = " ".join(words[:mid_point]) + "\n" + " ".join(words[mid_point:])
            new_labels.append(new_label)
        else:
            new_labels.append(label)
    return new_labels

def ring_graph(db, feature, feature_name, start_angle=40): 
    sizes = db[feature].value_counts().values.tolist()
    domain_categories = db[feature].value_counts().index.tolist()
    labels = add_line_breaks(domain_categories)
    
    
    if feature == "domain_class":
        domain_categories_to_skip = []
        
    if feature == "is_subjective":
        mapping = {"True": 'Subjective', "False": 'Objective'}
        domain_categories = [mapping[category] for category in domain_categories]


    # Use the seaborn library to get a palette of 30 colors from the Blues color map
    palette = sns.color_palette("Blues", 30)
    hex_colors = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b in palette]
    evenly_spaced_indices = [int(i * (len(hex_colors) / len(domain_categories))) for i in range(len(domain_categories))]
    selected_evenly_spaced_colors = [hex_colors[i] for i in evenly_spaced_indices]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    

    # Create a pie chart with a hole in the center (donut chart)
    wedges, texts= ax.pie(
        sizes, labels=labels, startangle=start_angle, wedgeprops=dict(width=0.3), pctdistance=0.55, colors=selected_evenly_spaced_colors
    )

    # Customize the plot
    #plt.setp(autotexts, size=10, weight="bold")
    # Add a central text
    ax.text(0, 0, f'CausalQuest\n{feature_name}', horizontalalignment='center', verticalalignment='center', fontsize=36)
    

    # Adjust label positions and add leader lines
    for i, (wedge, text) in enumerate(zip(wedges, texts)):
        if feature == "domain_class" and (text.get_text() in domain_categories_to_skip or text.get_text()==""):
            continue
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))

        # Adjust label position to avoid overlap
        horizontalalignment = 'left' if x > 0 else 'right'
        connectionstyle = "angle,angleA=0,angleB={}".format(angle)

        text.set_horizontalalignment(horizontalalignment)
        text.set_position((x * 1.3, y * 1.3))
        # set size of the text
        text.set_fontsize(font_size)
        
        # Draw the leader lines
        ax.plot([x, x * 1.1], [y, y * 1.1], color='gray', lw=0.75, linestyle='-', transform=ax.transData, clip_on=False)
        ax.plot([x * 1.1, x * 1.3], [y * 1.1, y * 1.3], color='gray', lw=0.75, linestyle='-', transform=ax.transData, clip_on=False)

    if save: 
        # save in the folder
        plt.savefig(plots_folder + f"{feature}_ring_chart.pdf", dpi=1200, bbox_inches='tight')
    # Show the plot
    plt.show()
    






def run_linguistic_analysis(df):
    question_words = ["what", "why", "how", "when", "who", "where", "which"]
    results = {
        True: {'word_count': Counter(), 'tokens': [], 'types': set(), 'lengths': [], 'multiple_question_words': 0},
        False: {'word_count': Counter(), 'tokens': [], 'types': set(), 'lengths': [], 'multiple_question_words': 0},
        'overall': {'word_count': Counter(), 'tokens': [], 'types': set(), 'lengths': [], 'multiple_question_words': 0}
    }

    for index, row in df.iterrows():
        summary = row['summary']
        is_causal = row['is_causal']
        words = summary.lower().split()
        found_question_word = False
        question_word_count = 0
        
        results[is_causal]['tokens'].extend(words)
        results[is_causal]['types'].update(words)
        results[is_causal]['lengths'].append(len(words))
        results['overall']['tokens'].extend(words)
        results['overall']['types'].update(words)
        results['overall']['lengths'].append(len(words))

        for word in words:
            clean_word = word.rstrip('?:!,.')
            if clean_word in question_words:
                results[is_causal]['word_count'][clean_word] += 1
                results['overall']['word_count'][clean_word] += 1
                found_question_word = True
                question_word_count += 1
        if not found_question_word:
            results[is_causal]['word_count']['no_question_word'] += 1
            results['overall']['word_count']['no_question_word'] += 1

        if question_word_count > 1:
            results[is_causal]['multiple_question_words'] += 1
            results['overall']['multiple_question_words'] += 1

    # Create DataFrame to store results
    metrics = []
    for key in results:
        metrics.append({
            'is_causal': key,
            'ttr': len(results[key]['types']) / len(results[key]['tokens']) if results[key]['tokens'] else 0,
            'average_length': sum(results[key]['lengths']) / len(results[key]['lengths']) if results[key]['lengths'] else 0,
            'total_length_in_tokens': len(results[key]['tokens']),
            'multiple_question_words': results[key]['multiple_question_words'],
            'vocabulary_size': len(results[key]['types']),
            **results[key]['word_count']
        })

    return pd.DataFrame(metrics)


