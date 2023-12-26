from datasets import load_dataset
from transformers import ViTImageProcessor
import torch


dataset_train = load_dataset("D:\\tiger-side-classification\ChtoZaLevEtotTigr-4\\train",  #путь к скаченным папкам
                             split='train',
                             verification_mode='all_checks'
                             )
dataset_val = load_dataset("D:\\tiger-side-classification\ChtoZaLevEtotTigr-4\\train",
                           split='train',
                           verification_mode='all_checks'
                           )
# dataset_test = load_dataset('D:\\tiger-side-classification\ChtoZaLevEtotTigr-4\\train',
#     split='train', # training dataset
#     verification_mode='all_checks'  # set to True if seeing splits Error)
# )



