import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from collections import defaultdict, Counter
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("📘 NLP Assignment - Menu Driven Program")

menu = st.sidebar.selectbox("Choose a Question", [
    "Question 1: Positional Index",
    "Question 2: Term-Document Matrix",
    "Question 3: Preprocessing & Edit Distance",
    "Question 4: Levenshtein Distance",
    "Question 5: POS Tagging with HMM",
    "Question 6: Word Sense Disambiguation"
])

# Common documents
docs = {
    "Doc1": "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester.",
    "Doc2": "I was a student. I have taken MDS472C."
}

doc1 = docs['Doc1']
doc2 = docs['Doc2']

if menu == "Question 1: Positional Index":
    st.header("Question 1: Positional Index")
    pos_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in docs.items():
        tokens = word_tokenize(text.lower())
        for i, word in enumerate(tokens):
            pos_index[word][doc_id].append(i)
    st.write("Positional Index:", dict(pos_index))

    query_words = st.text_input("Enter words to search positional indexes (comma separated)", "student, MDS472C")
    if query_words:
        for word in query_words.lower().split(','):
            st.write(f"Positions for '{word.strip()}':", pos_index.get(word.strip(), {}))

elif menu == "Question 2: Term-Document Matrix":
    st.header("Question 2: Term-Document Matrix")
    all_terms = sorted(set(word_tokenize(doc1.lower())) | set(word_tokenize(doc2.lower())))
    term_matrix = {term: [int(term in word_tokenize(doc1.lower())), int(term in word_tokenize(doc2.lower()))] for term in all_terms}
    st.table([['Term', 'Doc1', 'Doc2']] + [[term] + term_matrix[term] for term in term_matrix])

elif menu == "Question 3: Preprocessing & Edit Distance":
    st.header("Question 3: Text Preprocessing and Edit Distance")
    doc1 = st.text_area("Enter Document 1", doc1)
    doc2 = st.text_area("Enter Document 2", doc2)
    tokens1 = word_tokenize(doc1.lower())
    tokens2 = word_tokenize(doc2.lower())
    st.write("Tokens Doc1:", tokens1)
    st.write("Tokens Doc2:", tokens2)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    pre1 = [lemmatizer.lemmatize(stemmer.stem(w)) for w in tokens1]
    pre2 = [lemmatizer.lemmatize(stemmer.stem(w)) for w in tokens2]
    st.write("Preprocessed Doc1:", pre1)
    st.write("Preprocessed Doc2:", pre2)

    freq_index = Counter(pre1 + pre2)
    st.write("Frequency Index:", freq_index)

    sorted_freq = sorted(freq_index.items(), key=lambda x: (-x[1], x[0]))
    st.write("Sorted Frequency:", sorted_freq)

    w1 = st.text_input("Edit Distance Word 1", "student")
    w2 = st.text_input("Edit Distance Word 2", "studied")
    if w1 and w2:
        dist = edit_distance(w1, w2)
        st.write(f"Edit Distance between '{w1}' and '{w2}' is {dist}")

elif menu == "Question 4: Levenshtein Distance":
    st.header("Question 4: Levenshtein Distance")
    A = "characterization"
    B = "categorization"

    def compute_edit_matrix(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j-1]+cost, dp[i][j-1]+1, dp[i-1][j]+1)
        return dp

    def backtrace(dp, a, b):
        i, j = len(a), len(b)
        operations = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
                op = 'Match' if a[i-1] == b[j-1] else 'Substitute'
                operations.append((op, a[i-1], b[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                operations.append(('Insert', '-', b[j-1]))
                j -= 1
            else:
                operations.append(('Delete', a[i-1], '-'))
                i -= 1
        return operations[::-1]

    dp_matrix = compute_edit_matrix(A, B)
    trace = backtrace(dp_matrix, A, B)
    st.write("Edit Matrix:")
    st.write(np.array(dp_matrix))

    aligned_A, aligned_B, op_seq = '', '', ''
    ins = dels = subs = match = 0
    for op, a_c, b_c in trace:
        aligned_A += a_c
        aligned_B += b_c
        if op == 'Insert': ins += 1; op_seq += '*'
        elif op == 'Delete': dels += 1; op_seq += '*'
        elif op == 'Substitute': subs += 1; op_seq += 's'
        else: match += 1; op_seq += '-'

    st.write("Aligned A:", aligned_A)
    st.write("Aligned B:", aligned_B)
    st.write("Operations:", op_seq)
    st.write(f"Total Edit Distance: {dp_matrix[len(A)][len(B)]}, Insertions: {ins}, Deletions: {dels}, Substitutions: {subs}, Matches: {match}")

elif menu == "Question 5: POS Tagging with HMM":
    st.header("Question 5: POS Tagging using HMM")
    training_sents = [
        ("The cat chased the rat").lower().split(),
        ("A rat can run").lower().split(),
        ("The dog can chase the cat").lower().split()
    ]

    pos_tags = [
        ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],
        ['DET', 'NOUN', 'MODAL', 'VERB'],
        ['DET', 'NOUN', 'MODAL', 'VERB', 'DET', 'NOUN']
    ]

    tags = set()
    transition = defaultdict(lambda: defaultdict(int))
    emission = defaultdict(lambda: defaultdict(int))

    for sent, tags_seq in zip(training_sents, pos_tags):
        prev = '<s>'
        for word, tag in zip(sent, tags_seq):
            transition[prev][tag] += 1
            emission[tag][word] += 1
            tags.add(tag)
            prev = tag

    st.write("Transition Probabilities:", dict(transition))
    st.write("Emission Probabilities:", dict(emission))

    st.write("Test Sentence:")
    test_sent = "the rat can chase the cat".split()
    st.write(test_sent)
    st.write("Simplified HMM tagging:")

    simple_tags = []
    for word in test_sent:
        scores = {tag: emission[tag].get(word, 0) for tag in tags}
        best = max(scores, key=scores.get)
        simple_tags.append((word, best))

    st.write(simple_tags)

elif menu == "Question 6: Word Sense Disambiguation":
    st.header("Question 6: Lesk Algorithm for Word Sense Disambiguation")
    sentence = st.text_area("Enter a sentence for WSD", "I went to the bank to deposit money")
    tokens = word_tokenize(sentence)
    open_class_words = [word for word in tokens if wn.synsets(word)]
    st.write("Open Class Words:", open_class_words)

    from nltk.wsd import lesk
    senses = {}
    for word in open_class_words:
        sense = lesk(tokens, word)
        senses[word] = sense.definition() if sense else "No definition found"

    st.write("Disambiguated Senses:", senses)
