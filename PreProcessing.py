from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os

docred_rel2id = json.load(open('DocRED/DocRED_baseline_metadata/rel2id.json', 'r'))

