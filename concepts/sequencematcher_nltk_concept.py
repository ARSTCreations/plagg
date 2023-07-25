import nltk
from difflib import SequenceMatcher

nltk.download('punkt')

def calculate_text_similarity(text1, text2):
    similarity = SequenceMatcher(None, text1, text2).ratio()

    return similarity

text1 = r"Dengan pertumbuhan penggunaan smartphone yang terus meningkat dan popularitas yang kian meluas pada smartphone berbasis sistem operasi Android"
text2 = r"smartphone smartphone berbasis pertumbuhan yang kian operasi penggunaan terus pada dan meluas meningkat sistem yang Android popularitas Dengan"

similarity = calculate_text_similarity(text1, text2)
print("Similarity Score:", similarity)
