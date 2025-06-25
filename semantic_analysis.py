import spacy
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

def analyze_semantics(paragraph):
    """
    Perform semantic analysis on a paragraph and return various insights.
    
    Args:
        paragraph (str): The input paragraph to analyze
        
    Returns:
        dict: A dictionary containing various semantic analysis results
    """
    results = {}
    
    # Process the text with spaCy
    doc = nlp(paragraph)
    
    # 1. Sentiment Analysis
    # Using TextBlob
    blob = TextBlob(paragraph)
    results['sentiment_textblob'] = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }
    
    # Using NLTK's Vader
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(paragraph)
    results['sentiment_vader'] = vader_scores
    
    # 2. Named Entity Recognition
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    results['named_entities'] = dict(entities)
    
    # 3. Part-of-Speech Tagging
    pos_tags = defaultdict(list)
    for token in doc:
        pos_tags[token.pos_].append(token.text)
    results['pos_tags'] = dict(pos_tags)
    
    # 4. Lemmatization
    lemmas = [token.lemma_ for token in doc]
    results['lemmas'] = lemmas
    
    # 5. Dependency Parsing (simplified)
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    results['dependencies'] = dependencies
    
    # 6. Key Phrases (noun chunks)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    results['key_phrases'] = key_phrases
    
    # 7. Word Frequency
    word_freq = defaultdict(int)
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            word_freq[token.lemma_.lower()] += 1
    results['word_frequency'] = dict(word_freq)
    
    # 8. Semantic Similarity (example between sentences)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) > 1:
        # Compare first two sentences as an example
        sent1 = nlp(sentences[0])
        sent2 = nlp(sentences[1])
        results['semantic_similarity'] = sent1.similarity(sent2)
    
    return results

def print_semantic_analysis(results):
    """Pretty print the semantic analysis results"""
    print("\n=== Semantic Analysis Results ===")
    
    # Sentiment
    print("\n1. Sentiment Analysis:")
    print(f"TextBlob - Polarity: {results['sentiment_textblob']['polarity']:.2f} (positive if >0)")
    print(f"TextBlob - Subjectivity: {results['sentiment_textblob']['subjectivity']:.2f}")
    print(f"VADER - Positive: {results['sentiment_vader']['pos']:.2f}")
    print(f"VADER - Negative: {results['sentiment_vader']['neg']:.2f}")
    print(f"VADER - Neutral: {results['sentiment_vader']['neu']:.2f}")
    print(f"VADER - Compound: {results['sentiment_vader']['compound']:.2f}")
    
    # Named Entities
    print("\n2. Named Entities:")
    for entity_type, entities in results['named_entities'].items():
        print(f"{entity_type}: {', '.join(entities)}")
    
    # Key Phrases
    print("\n3. Key Phrases (noun chunks):")
    print(", ".join(results['key_phrases']))
    
    # Word Frequency
    print("\n4. Most Frequent Words (excluding stop words):")
    sorted_words = sorted(results['word_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]
    for word, freq in sorted_words:
        print(f"{word}: {freq}")
    
    # Semantic Similarity (if available)
    if 'semantic_similarity' in results:
        print(f"\n5. Semantic Similarity between first two sentences: {results['semantic_similarity']:.2f}")

# Example usage
if __name__ == "__main__":
    example_paragraph = """
    Football is a wonderful game.
    I love playing football.
    Football is my favorite sport.
    Football is a great way to stay fit.
    There is a lot of bragging in football.
    """
    
    analysis_results = analyze_semantics(example_paragraph)
    print_semantic_analysis(analysis_results)