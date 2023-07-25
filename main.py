import os
import psutil
import re
import requests
import urllib3
import signal
import time
import nltk
import warnings
print("1",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from dotenv import dotenv_values
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import tensorflow_hub as hub
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import abort
from waitress import serve
print("2",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
nltk.download('punkt', quiet=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)
print("3",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
try:
    embed = hub.load("use-4")
    use4 = True
except Exception:
    print("Model Not Found, USE4 feature will be disabled")
    use4 = False
print("4",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


print("Program is starting...")

ENV_VARS = dotenv_values()
API_KEY = ENV_VARS['API_KEY']
CSE_SID = ENV_VARS['CSE_SID']

def signal_handler(sig, frame):
    print('Aborting...')
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

def __sanitize(search_term):
    sanitized_term = re.sub(r'[^a-zA-Z0-9\s]', '', search_term)
    return sanitized_term.strip()

def __clean_content(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.replace('\n', ' ')
    return cleaned_string

def __google_search(search_term, API_KEY, CSE_SID, lang):
    urls = []
    url = "https://www.googleapis.com/customsearch/v1?key="+API_KEY+"&cx="+CSE_SID+"&q="+search_term+"&lr="+lang
    data = requests.get(url).json()
    search_items = data.get("items")
    urls = [search_item.get("link") for search_item in search_items] if search_items != None else []

    clean_list = list(set(urls))
    print("Total URLs: ", len(clean_list))
    return clean_list

def __seq_match_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def __tfidf_similarity(text1, text2):
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

def __use4_similarity(text1, text2, individual_embedding=False):
    if embed != None:
        if individual_embedding:
            embeddings1 = embed([text1]).numpy()
            embeddings2 = embed([text2]).numpy()
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        else:
            embeddings = embed([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    else:
        print("USE4 feature is disabled")
        raise Exception("USE4 feature is disabled")

    return similarity

def __word2vec_similarity(text1, text2):
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    model = Word2Vec([tokens1, tokens2], min_count=1, vector_size=100)

    vector1 = sum([model.wv[word] for word in tokens1]) / len(tokens1)
    vector2 = sum([model.wv[word] for word in tokens2]) / len(tokens2)

    similarity = cosine_similarity([vector1], [vector2])[0][0]

    return similarity

def __get_js_enabled(url):
    driver.get(url)
    driver.implicitly_wait(15)
    page_content = driver.page_source
    driver.quit()
    return page_content

def __compare(text, url, rto=100, model='word2vec'):
    response = requests.get(url, timeout=rto, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding="iso-8859-1")
    souped = soup.get_text()
    webpage_text = __sanitize(__clean_content(souped))

    def similarity(text1, text2, model):
        if model == 'tfidf':
            print("Using TFIDF")
            return (round(__tfidf_similarity(text1, text2)*100,2))
        elif model == 'use4':
            print("Using USE4")
            return (round(__use4_similarity(text1, text2)*100,2))
        elif model == 'seq_match':
            print("Using Sequence Matcher")
            return (round(__seq_match_similarity(text1, text2)*100,2))
        else:
            print("Using Word2Vec")
            return (round(__word2vec_similarity(text1, text2)*100,2))

    if len(re. findall(r'\w+', webpage_text)) < 30:
        print("Checking Web Using JS")
        response = __get_js_enabled(url)
        soup = BeautifulSoup(response, 'html.parser', from_encoding="iso-8859-1")
        souped = soup.get_text()
        webpage_text = __sanitize(__clean_content(souped))
    else:
        print("Web Checked Using Requests Only")

    similarity = similarity(text, webpage_text, model)

    return similarity, webpage_text

def process(srch_query, similarity_threshold=20, model='word2vec'):
    lang_focus = detect(srch_query)
    srch_array = __google_search(srch_query, API_KEY, CSE_SID, lang_focus)

    percentages = []
    failed_urls = []
    strong_urls = []
    for url in srch_array:
        try:
            print("Checking: ", url)
            similarity_ratio, webpage_text = __compare(srch_query, url, 10, model)
            if similarity_ratio >= similarity_threshold:
                strong_urls.append([url, similarity_ratio, webpage_text])
        except:
            similarity_ratio = -1
        percentages.append(similarity_ratio) if similarity_ratio != -1 else failed_urls.append(url)

    if len(percentages) != 0:
        return max(percentages), round(sum(percentages)/len(percentages),2), min(list(filter(lambda x: x != 0, percentages))), strong_urls, model
    else:
        return -1,-1,-1, strong_urls

@app.route('/process', methods=['GET', 'POST'])
def post_get_process():
    time_start = time.time()
    text = request.args.get('text') or request.form.get('text')
    model = request.args.get('model') or request.form.get('model') or 'word2vec'
    
    if text == None or text == '':
        abort(400, BaseException('Error: Text not provided.'))
    max_percent, avg_percent, min_percent, strong_urls, model_used = process(text, model=model)
    # try:
    #     max_percent, avg_percent, min_percent, strong_urls = process(text)
    # except:
    #     abort(500)
    time_end = time.time()
    return jsonify({'status_code': '200', 
                    'message': 'Success: Text processed.',
                    'max_percent': max_percent,
                    'avg_percent': avg_percent,
                    'min_percent': min_percent,
                    'strong_urls': strong_urls,
                    'time_taken': round(time_end-time_start,2),
                    'model': model_used})

@app.errorhandler(400)
def page_not_found(e):
    return jsonify({'status_code': '400',
                    'message': str(e)}), 400

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'status_code': '404',
                    'message': 'Error: Page not found.'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'status_code': '500',
                    'message': 'Error: Internal server error.'}), 500
@app.route('/', use4=False)
def index():
    title = "Plagg API v1.0"
    return render_template('index.html', title=title)

print("5",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# Waitress server
# serve(app, host='0.0.0.0', port=3000)
# Werkzeug server
app.run(host='0.0.0.0', port=3000, debug=True)