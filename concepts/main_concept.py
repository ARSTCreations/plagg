import re
import nltk
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from googlesearch import search
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt', quiet=True)

def clean_content(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.replace('\n', '')
    return cleaned_string

def calculate_text_similarity(text1, text2):
    if len(text1) > len(text2):
        longer_text = text1
        shorter_text = text2
    else:
        longer_text = text2
        shorter_text = text1

    vectorizer = TfidfVectorizer()
    tfidf_features_longer = vectorizer.fit_transform([longer_text])
    tfidf_features_shorter = vectorizer.transform([shorter_text])
    similarity_matrix = cosine_similarity(tfidf_features_longer, tfidf_features_shorter)

    return similarity_matrix[0][0]

def is_duplicate_text(text, url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    souped = soup.get_text()
    webpage_text = clean_content(souped)
    # print("\n Content of the webpage: \n", webpage_text)

    similarity = calculate_text_similarity(text, webpage_text)

    similarity_threshold = 0.8
    if similarity >= similarity_threshold:
        return round(similarity*100,2)
    else:
        return round(similarity*100,2)

srch_query = clean_content(r'intext:"'+input("Enter your search query: ")+'"')
# srch_query = input("Enter your search query: ")
# print(srch_query, "\nSearching...")
print("Searching...")
lang_focus = detect(srch_query)
# print("Language detected: ", lang_focus)
# srch_resul = search(srch_query, num=20, stop=20, pause=4)
srch_resul = search(srch_query, lang=lang_focus, num=20, stop=20, pause=2)
srch_array = list(srch_resul)
# print(srch_array)

firing_urls = []
percentages = []
for url in srch_array:
    try:
        similarity_ratio = is_duplicate_text(srch_query, url)
    except:
        print("Error in URL: ", url)
        similarity_ratio = -1
    print(url, similarity_ratio)
    if similarity_ratio != -1:
        firing_urls.append(url)
        percentages.append(similarity_ratio)

if len(percentages) != 0:
    print("Avg Similarity: "+str(round(sum(percentages)/len(percentages),2)))
    print("Max Similarity: "+str(max(percentages)))
    print("Min Similarity: "+str(min(percentages)))
    print("Min (non zero) Similarity: "+str(min(list(filter(lambda x: x != 0, percentages)))))