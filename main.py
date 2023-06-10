import re
import signal
import nltk
import requests
import urllib3
from bs4 import BeautifulSoup
from dotenv import dotenv_values
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt', quiet=True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

env_vars = dotenv_values()
api_key = env_vars['api_key']
cse_sid = env_vars['cse_sid']

def signal_handler(sig, frame):
    print('Skipped!!!')
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

def sanitize(search_term):
    sanitized_term = re.sub(r'[^a-zA-Z0-9\s]', '', search_term)
    return sanitized_term.strip()

def clean_content(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.replace('\n', '')
    return cleaned_string

def google_search(search_term, api_key, cse_sid, lang):
    urls = []
    for s in range(1,3):
        url = "https://www.googleapis.com/customsearch/v1?key="+api_key+"&cx="+cse_sid+"&q="+search_term+"&start="+str(s)+"&lr="+lang
        data = requests.get(url).json()
        search_items = data.get("items")
        links = [search_item.get("link") for i, search_item in enumerate(search_items, start=s)]
        urls = urls + links
    return urls

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

def journalist_write(file_name, url, similarity, webpage_text):
    file = open(file_name, "a+")
    write_string = url+"\n"+str(round(similarity*100,2))+"%\n"+webpage_text+"\n\n"
    file.write(write_string)
    file.close()

def compare(text, url, rto=100):
    response = requests.get(url, timeout=rto, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding="iso-8859-1")
    souped = soup.get_text()
    webpage_text = sanitize(clean_content(souped))

    similarity = calculate_text_similarity(text, webpage_text)

    similarity_threshold = 0.4
    if similarity >= similarity_threshold:
        journalist_write("fatal_detections.txt", url, similarity, webpage_text)
        return round(similarity*100,2)
    else:
        return round(similarity*100,2)

srch_query = sanitize(clean_content(input("Enter your search query: ")))
print("\nSearching top 20...")
lang_focus = detect(srch_query)
srch_array = google_search(srch_query, api_key, cse_sid, lang_focus)

percentages = []
failed_urls = []
for count,url in enumerate(srch_array, 1):
    try:
        similarity_ratio = compare(srch_query, url)
    except:
        similarity_ratio = -1
    print(str(count)+". "+url+" ==> "+str(similarity_ratio)+"%")
    percentages.append(similarity_ratio) if similarity_ratio != -1 else failed_urls.append(url)

if len(percentages) != 0:
    print("Max Similarity: "+str(max(percentages))+"%")
    print("Avg Similarity: "+str(round(sum(percentages)/len(percentages),2))+"%")
    print("Min Similarity: "+str(min(list(filter(lambda x: x != 0, percentages))))+"%")