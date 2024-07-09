import os
import openai
import json
import numpy as np
import re
import string
from collections import Counter

#Metrics calculation functions
#Function that removes innecesary strings in the answer or gt
def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  def white_space_fix(text):
      return " ".join(text.split())
  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
      return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

#Origial ReACT function that calculates the f1_score
def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall

#Metric that calculates F1 score and Exact Match. Just need one string as the prediction and
# it´s respective ground truth
def get_metrics(pred, gt):
  if pred is not None:
    em = (pred == gt)
    f1 = f1_score(pred, gt)[0]
    return {'em': int(em), 'f1': f1}
  return {'em': 0, 'f1': 0}