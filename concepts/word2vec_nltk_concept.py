import nltk
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

nltk.download('punkt')

def calculate_text_similarity(text1, text2):
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    model = Word2Vec([tokens1, tokens2], min_count=1, vector_size=100)

    vector1 = sum([model.wv[word] for word in tokens1]) / len(tokens1)
    vector2 = sum([model.wv[word] for word in tokens2]) / len(tokens2)

    similarity = cosine_similarity([vector1], [vector2])[0][0]

    return similarity

text1 = r"Dengan pertumbuhan penggunaan smartphone yang terus meningkat dan popularitas yang kian meluas pada smartphone berbasis sistem operasi Android"
text2 = r"smartphone smartphone berbasis pertumbuhan yang kian operasi penggunaan terus pada dan meluas meningkat sistem yang Android popularitas Dengan"

similarity = calculate_text_similarity(text1, text2)
print("Similarity Score:", similarity)
