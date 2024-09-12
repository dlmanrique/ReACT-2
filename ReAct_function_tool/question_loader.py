import json
import numpy as np
import pandas as pd
from utils import *


DATA_DIR = "data"

class QuestionLoader():
  def __init__(self):
    super().__init__()
    data_file = f"{DATA_DIR}/{'baseline_gh.csv'}"
    self.data = pd.read_csv(data_file, encoding='utf-8')
    #Remove Single Hop line
    #self.data = self.data.drop(10)
    self.data.columns = ["Creador", "Pregunta", "Respuesta"]


  def load_question(self, idx):
    question = self.data.iloc[idx]["Pregunta"]
    return question

  def get_gt(self, idx):
    gt = self.data.iloc[idx]["Respuesta"]
    return gt
  
  def __len__(self):
    return len(self.data)