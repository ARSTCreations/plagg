import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # Download necessary tokenizer data (only required once)

def calculate_text_similarity(text1, text2):
    # Determine the longer and shorter texts
    if len(text1) > len(text2):
        longer_text = text1
        shorter_text = text2
    else:
        longer_text = text2
        shorter_text = text1

    # Tokenize the texts
    tokens = nltk.word_tokenize(longer_text + ' ' + shorter_text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF features for the longer text
    tfidf_features_longer = vectorizer.fit_transform([longer_text])

    # Transform the shorter text using the same vectorizer
    tfidf_features_shorter = vectorizer.transform([shorter_text])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_features_longer, tfidf_features_shorter)

    # Extract significant features
    feature_names = vectorizer.vocabulary_
    feature_weights = tfidf_features_longer.toarray()[0]
    significant_features = [(feature, feature_weights[feature_names[feature]]) for feature in feature_names]

    return similarity_matrix[0][0], significant_features

# Example usage:
text1 = "Lorem ipsum dolor sit amet, Nulla varius vestibulum nunc, id dapibus libero vulputate vitae."
text2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam aliquet efficitur libero in porttitor. Nunc a iaculis erat, sit amet volutpat urna. In eget ultrices nisi, id mollis sapien. Nullam eget velit at velit vulputate euismod nec ut augue. Nam vitae quam leo. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla varius vestibulum nunc, id dapibus libero vulputate vitae. Nullam quis commodo libero. Nunc lacinia, nisi nec blandit volutpat, lorem magna consequat velit, non interdum ante erat in neque. Morbi efficitur vel neque sit amet commodo."

similarity, significant_features = calculate_text_similarity(text1, text2)
print("Similarity Score:", similarity)
print("Top 5 Significant Features:")
# top_5_features = sorted(significant_features, key=lambda x: x[1], reverse=True)[:5]
top_5_features = sorted(significant_features, key=lambda x: x[1], reverse=True)
for feature, weight in top_5_features:
    # print(f"- {feature}: {weight}")
    print("- "+feature + ": "+str(round(weight*100,2))+"%")
