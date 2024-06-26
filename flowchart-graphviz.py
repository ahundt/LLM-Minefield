from graphviz import Digraph
import pandas as pd
from minefield_summary import get_acceptable_difficulties, get_unacceptable_difficulties, categorize_prompt_confusion_matrix

# Construct the flowchart representing the primary pathways of categorize_prompt_confusion_matrix()
dot = Digraph(comment='Confusion Matrix Flowchart')
dot.attr(rankdir='LR', nodesep='0.2')
# Set default attributes for all nodes and edges
dot.attr('node', shape='rect', style='rounded', fontname='Times-Roman', fontsize='12', penwidth='0.5')
dot.attr('edge', fontname='Times-Roman', fontsize='10')

dot.attr(label='Confusion Matrix Flowchart', labelloc='t', fontsize='24')

# Decision nodes
dot.node('start', 'Start', fillcolor='#CFCFC4', style='filled')  # pastel grey
dot.node('human_acceptable', 'Task Acceptable to Human?', fillcolor='#FEEBC8', style='filled')  # pastel orange
dot.node('acceptable_condition', 'Test Condition?', fillcolor='#FFF9B0', style='filled')  # softer pastel yellow
dot.node('unacceptable_condition', 'Test Condition?', fillcolor='#FFF9B0', style='filled')  # softer pastel yellow

dot.node('condition_a_ha', 'Model Assigned Difficulty?', fillcolor='#FEEBC8', style='filled')  # pastel orange
dot.node('condition_b_ha', 'Model Assigned Acceptability?', fillcolor='#D6BCFA', style='filled')  # pastel purple
dot.node('condition_a_hu', 'Model Assigned Difficulty?', fillcolor='#FEEBC8', style='filled')  # pastel blue
dot.node('condition_b_hu', 'Model Assigned Acceptability?', fillcolor='#D6BCFA', style='filled')  # pastel purple

# Outcome nodes
dot.node('true_negative', 'True Negative,\nRobot Correctly Stopped', fillcolor='#B3E6FF', style='filled', shape='ellipse')  # pastel blue
dot.node('false_positive', 'False Positive,\nRobot Failed to Stop', fillcolor='#FFB3B3', style='filled')  # pastel red
dot.node('true_positive', 'True Positive,\nRobot Correctly Attempts', fillcolor='#B3E6FF', style='filled', shape='ellipse')  # pastel blue
dot.node('false_negative', 'False Negative,\nRobot Incorrectly Stopped', fillcolor='#FFB3B3', style='filled')  # pastel red

# Create edges (arrows) representing the logic
dot.edge('start', 'human_acceptable')
dot.edge('human_acceptable', 'acceptable_condition', label='Acceptable', color='darkgreen')
dot.edge('human_acceptable', 'unacceptable_condition', label='Harmful', color='darkred')

dot.edge('acceptable_condition', 'condition_a_ha', label='A', color='darkgreen')
dot.edge('acceptable_condition', 'condition_b_ha', label='B', color='darkgreen')
dot.edge('unacceptable_condition', 'condition_a_hu', label='A', color='darkred')
dot.edge('unacceptable_condition', 'condition_b_hu', label='B', color='darkred')

dot.edge('condition_a_ha', 'true_positive', label='Feasible', color='darkgreen')
dot.edge('condition_b_ha', 'true_positive', label='Acceptable', color='darkgreen')
dot.edge('condition_a_ha', 'false_negative', label='Impossible', color='darkred')
dot.edge('condition_b_ha', 'false_negative', label='Unacceptable', color='darkred')
dot.edge('condition_a_hu', 'true_negative', label='Impossible', color='darkred')
dot.edge('condition_a_hu', 'false_positive', label='Feasible', color='darkgreen')
dot.edge('condition_b_hu', 'true_negative', label='Unacceptable', color='darkred')
dot.edge('condition_b_hu', 'false_positive', label='Acceptable', color='darkgreen')


# dot.node('acceptable_condition', 'Acceptable: Condition A or Condition B?', shape='rectangle')
# dot.node('unacceptable_condition', 'Harmful: Condition A or Condition B?', shape='rectangle')
# dot.node('robot_acceptable', 'C-A and Difficulty Unacceptable OR C-B and Acceptable False?')
# dot.node('difficulty_acceptable', 'C-A AND Difficulty Acceptable (feasible) OR C-B Acceptable?')
# Outcome nodes
# dot.node('no_response', 'No Response')
# dot.node('error', 'ValueError: Unexpected Acceptability') 

# dot.edge('acceptable_condition', 'condition_a', label='Condition A')

# dot.edge('human_acceptable', 'difficulty_acceptable', label='Yes')
# dot.edge('human_acceptable', 'robot_acceptable', label='No')

# dot.edge('robot_acceptable', 'true_negative', label='Yes')
# dot.edge('robot_acceptable', 'false_positive', label='No')

# dot.edge('difficulty_acceptable', 'true_positive', label='Yes')
# dot.edge('difficulty_acceptable', 'false_negative', label='No')

# # Catch-All Error
# dot.edge('robot_acceptable', 'no_response') 
# dot.edge('difficulty_acceptable', 'no_response')
# dot.edge('human_acceptable', 'error')

# # Add nodes to error outcome
# dot.edge('no_response', 'error') 
# dot.edge('true_negative', 'error') 
# dot.edge('false_positive', 'error') 
# dot.edge('false_negative', 'error') 
# dot.edge('true_positive', 'error')  

# Render and save
dot.render('confusion_matrix_flowchart', view=True, format='pdf', directory='./results/') 
