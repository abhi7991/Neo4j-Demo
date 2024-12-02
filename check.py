import streamlit as st
import requests
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta
import pandas as pd
from modules import utils
import numpy as np
from openai import OpenAI
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

utils.graph_intit()
