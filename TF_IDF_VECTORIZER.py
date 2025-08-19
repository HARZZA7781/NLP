import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import Counter
from itertools import combinations
import math
import io

# -------------------------
# Q1: Cosine similarity & word analogies
# -------------------------
def q1_cosine_analogy(docs, target_word, analogy_words):
    # Check if inputs are provided
    if not docs.strip():
        return None, "Please provide documents", None, None
    
    docs_list = [d.strip() for d in docs.split("\n") if d.strip()]
    
    if len(docs_list) < 2:
        return None, "Please provide at least 2 documents", None, None
    
    try:
        # TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_list)
        tfidf_words = tfidf_vectorizer.get_feature_names_out()
        tfidf_vectors = tfidf_matrix.T.toarray()
        
        cos_sim_matrix = cosine_similarity(tfidf_vectors)
        cos_sim_df = pd.DataFrame(cos_sim_matrix, index=tfidf_words, columns=tfidf_words)
        
        # Nearest words function
        def nearest_words(word, top_n=5):
            if word not in tfidf_words:
                return f"Word '{word}' not found in vocabulary"
            idx = np.where(tfidf_words == word)[0][0]
            sim_scores = list(enumerate(cos_sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            nearest = [(tfidf_words[i], round(score, 3)) for i, score in sim_scores[1:top_n+1]]
            return nearest
        
        # Find nearest words for target word
        nearest_result = nearest_words(target_word) if target_word else "No target word provided"
        
        # Analogy computation
        analogy_result = None
        if analogy_words and analogy_words.strip():
            try:
                word_parts = analogy_words.lower().strip().split()
                if len(word_parts) != 3:
                    analogy_result = "Please provide exactly 3 words for analogy: wordA wordB wordC"
                else:
                    word_a, word_b, word_c = word_parts
                    if all(w in tfidf_words for w in [word_a, word_b, word_c]):
                        vec_a = tfidf_vectors[np.where(tfidf_words == word_a)[0][0]]
                        vec_b = tfidf_vectors[np.where(tfidf_words == word_b)[0][0]]
                        vec_c = tfidf_vectors[np.where(tfidf_words == word_c)[0][0]]
                        target_vec = vec_b - vec_a + vec_c
                        similarities = cosine_similarity([target_vec], tfidf_vectors)[0]
                        best_idx = similarities.argsort()[::-1]
                        results = [tfidf_words[i] for i in best_idx if tfidf_words[i] not in [word_a, word_b, word_c]]
                        analogy_result = f"Best match: {results[0]}" if results else "No good match found"
                    else:
                        missing_words = [w for w in [word_a, word_b, word_c] if w not in tfidf_words]
                        analogy_result = f"Words not in vocabulary: {missing_words}"
            except Exception as e:
                analogy_result = f"Error in analogy computation: {str(e)}"
        
        # Create plot
        plot_buf = None
        if target_word and target_word in tfidf_words:
            try:
                nearest_words_list = nearest_words(target_word)
                if isinstance(nearest_words_list, list) and len(nearest_words_list) > 0:
                    words_to_plot = [target_word] + [w for w, _ in nearest_words_list[:4]]  # Limit to prevent overcrowding
                    indices = [np.where(tfidf_words == w)[0][0] for w in words_to_plot if w in tfidf_words]
                    
                    if len(indices) > 1:
                        pca = PCA(n_components=2)
                        reduced_vectors = pca.fit_transform(tfidf_vectors)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        colors = ['red'] + ['blue'] * (len(indices) - 1)  # Target word in red, others in blue
                        
                        for i, idx in enumerate(indices):
                            ax.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], 
                                     s=100, c=colors[i], alpha=0.7)
                            ax.annotate(tfidf_words[idx], 
                                      (reduced_vectors[idx, 0], reduced_vectors[idx, 1]),
                                      xytext=(5, 5), textcoords='offset points', fontsize=10)
                        
                        ax.set_title(f"Nearest words to '{target_word}' (PCA visualization)")
                        ax.set_xlabel('First Principal Component')
                        ax.set_ylabel('Second Principal Component')
                        ax.grid(True, alpha=0.3)
                        
                        plot_buf = io.BytesIO()
                        plt.savefig(plot_buf, format="png", dpi=150, bbox_inches='tight')
                        plt.close()
                        plot_buf.seek(0)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
        
        return cos_sim_df.round(3), nearest_result, analogy_result, plot_buf
        
    except Exception as e:
        return None, f"Error processing documents: {str(e)}", None, None


# -------------------------
# Q2: TF-IDF & Euclidean normalization (Auto-calculate everything)
# -------------------------
def q2_auto_tfidf(documents, query):
    if not documents.strip():
        return None, "Please provide documents", None, None, None, None
    
    try:
        # Parse documents
        docs_list = [d.strip() for d in documents.split("\n") if d.strip()]
        if len(docs_list) < 2:
            return None, "Please provide at least 2 documents", None, None, None, None
        
        # Use sklearn's TfidfVectorizer to compute everything automatically
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_list)
        
        # Get feature names (vocabulary)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Create TF-IDF DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=feature_names,
                               index=[f'Doc{i+1}' for i in range(len(docs_list))])
        
        # Calculate raw term frequencies
        count_vectorizer = CountVectorizer(lowercase=True, vocabulary=feature_names)
        tf_matrix = count_vectorizer.fit_transform(docs_list)
        tf_df = pd.DataFrame(tf_matrix.toarray(), 
                            columns=feature_names,
                            index=[f'Doc{i+1}' for i in range(len(docs_list))])
        
        # Calculate IDF values manually for display
        N = len(docs_list)  # Total number of documents
        idf_values = {}
        for term in feature_names:
            # Count documents containing the term
            df_t = sum(1 for doc in docs_list if term in doc.lower().split())
            if df_t > 0:
                idf_values[term] = math.log(N / df_t)
            else:
                idf_values[term] = 0
        
        # Create IDF DataFrame
        idf_df = pd.DataFrame([idf_values], index=['IDF'])
        
        # Query processing
        cos_df = None
        query_tfidf = None
        if query and query.strip():
            # Transform query using the same vectorizer
            query_vec = tfidf_vectorizer.transform([query.lower()])
            query_tfidf = pd.DataFrame(query_vec.toarray(), 
                                     columns=feature_names, 
                                     index=['Query'])
            
            # Compute cosine similarity
            cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            cos_df = pd.DataFrame({
                'Document': [f'Doc{i+1}' for i in range(len(docs_list))], 
                'Cosine Similarity': cos_scores
            }).sort_values(by='Cosine Similarity', ascending=False)
        
        # Euclidean normalization of TF values
        norm_tf_df = tf_df.copy()
        for doc_idx in norm_tf_df.index:
            norm = np.sqrt(np.sum(norm_tf_df.loc[doc_idx] ** 2))
            if norm > 0:
                norm_tf_df.loc[doc_idx] = norm_tf_df.loc[doc_idx] / norm
        
        # Round values for better display
        tfidf_df = tfidf_df.round(4)
        idf_df = idf_df.round(4)
        norm_tf_df = norm_tf_df.round(4)
        if query_tfidf is not None:
            query_tfidf = query_tfidf.round(4)
        
        return tf_df, tfidf_df, idf_df, cos_df, norm_tf_df, query_tfidf
        
    except Exception as e:
        return None, f"Error processing documents: {str(e)}", None, None, None, None


# -------------------------
# Q3: PMI
# -------------------------
def q3_pmi(documents, target_word):
    if not documents.strip():
        return None, "Please provide documents"
    
    try:
        corpus = [d.strip() for d in documents.split("\n") if d.strip()]
        if len(corpus) < 2:
            return None, "Please provide at least 2 documents"
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in corpus]
        
        # Count words and pairs
        word_counts = Counter([word for doc in tokenized_docs for word in doc])
        total_words = sum(word_counts.values())
        
        # Count co-occurrences (words appearing in the same document)
        pair_counts = Counter()
        for doc in tokenized_docs:
            unique_words = set(doc)
            for w1, w2 in combinations(unique_words, 2):
                pair_counts[tuple(sorted([w1, w2]))] += 1
        
        vocab = sorted(word_counts.keys())
        
        # Limit vocabulary size for display purposes
        if len(vocab) > 20:
            # Keep most frequent words
            most_frequent = [word for word, count in word_counts.most_common(20)]
            vocab = sorted(most_frequent)
        
        pmi_matrix = pd.DataFrame(index=vocab, columns=vocab, data=0.0)
        
        def compute_pmi(w1, w2):
            if w1 == w2:
                return 0.0
            
            p_w1 = word_counts[w1] / total_words
            p_w2 = word_counts[w2] / total_words
            p_w1_w2 = pair_counts.get(tuple(sorted([w1, w2])), 0) / len(tokenized_docs)
            
            if p_w1_w2 == 0 or p_w1 == 0 or p_w2 == 0:
                return 0.0
            
            return math.log2(p_w1_w2 / (p_w1 * p_w2))
        
        # Fill PMI matrix
        for i, w1 in enumerate(vocab):
            for j, w2 in enumerate(vocab):
                if i != j:
                    pmi_score = compute_pmi(w1, w2)
                    pmi_matrix.loc[w1, w2] = pmi_score
        
        # Target word similarities
        target_sim = None
        if target_word and target_word.strip():
            target_word = target_word.lower()
            if target_word in vocab:
                target_sim = pmi_matrix.loc[target_word].sort_values(ascending=False)
                target_sim = target_sim[target_sim != 0]  # Remove zero values
            else:
                target_sim = f"Word '{target_word}' not found in vocabulary"
        
        return pmi_matrix.round(3), target_sim
        
    except Exception as e:
        return None, f"Error computing PMI: {str(e)}"


# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 NLP Toolkit — Cosine Similarity, TF-IDF, PMI")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Q1: Cosine Similarity & Analogy", "📈 Q2: TF-IDF & Normalization", "🔗 Q3: PMI"])

with tab1:
    st.header("Cosine Similarity & Word Analogies")
    st.markdown("Enter documents and explore word relationships using cosine similarity.")
    
    docs_q1 = st.text_area("📝 Enter Documents (one per line)", height=150, key="docs_q1",
                          placeholder="The cat sat on the mat\nThe dog ran in the park\nCats and dogs are pets")
    
    col1, col2 = st.columns(2)
    with col1:
        target_word_q1 = st.text_input("🎯 Target word for similarity", key="target_q1", 
                                      placeholder="cat")
    with col2:
        analogy_words_q1 = st.text_input("🔄 Analogy (format: wordA wordB wordC)", key="analogy_q1",
                                        placeholder="king man woman")
    
    if st.button("🚀 Run Analysis", key="run_q1"):
        with st.spinner("Processing documents..."):
            cos_df, nearest, analogy_res, plot_buf = q1_cosine_analogy(docs_q1, target_word_q1, analogy_words_q1)
            
            if cos_df is not None:
                st.success("✅ Analysis completed!")
                
                st.subheader("📊 Cosine Similarity Matrix")
                st.dataframe(cos_df, use_container_width=True)
                
                st.subheader("🎯 Nearest Words")
                if isinstance(nearest, list):
                    if len(nearest) > 0:
                        nearest_df = pd.DataFrame(nearest, columns=['Word', 'Similarity Score'])
                        st.dataframe(nearest_df, use_container_width=True)
                    else:
                        st.info("No similar words found.")
                else:
                    st.warning(str(nearest))
                
                st.subheader("🔄 Analogy Result")
                if analogy_res:
                    st.info(analogy_res)
                else:
                    st.info("No analogy computed.")
                
                if plot_buf:
                    st.subheader("📈 Visualization")
                    st.image(plot_buf)
            else:
                st.error(f"❌ {nearest}")

with tab2:
    st.header("TF-IDF & Normalization")
    st.markdown("Automatically compute TF-IDF values from documents - just provide documents and query!")
    
    st.subheader("📝 Document Input")
    docs_q2 = st.text_area("📝 Enter Documents (one per line)", height=150, key="docs_q2",
                          placeholder="The best car insurance provides comprehensive coverage\nAuto insurance is essential for every car owner\nBest auto deals available at the car dealership\nCar insurance rates depend on your driving record\nInsurance companies offer the best protection plans")
    
    query_q2 = st.text_input("🔍 Query", key="query_q2", 
                            placeholder="car insurance",
                            help="Enter your search query to rank documents")
    
    if st.button("🚀 Run Analysis", key="run_q2"):
        with st.spinner("Computing TF-IDF automatically..."):
            result = q2_auto_tfidf(docs_q2, query_q2)
            
            if result[0] is not None:
                tf_df, tfidf_df, idf_df, cos_df, norm_tf_df, query_tfidf = result
                st.success("✅ Analysis completed!")
                
                # Show only top 10 most important terms to avoid clutter
                important_terms = tfidf_df.sum().nlargest(10).index.tolist()
                
                st.subheader("📊 Term Frequency (TF) Matrix")
                st.markdown("*Raw count of terms in each document*")
                st.dataframe(tf_df[important_terms], use_container_width=True)
                
                st.subheader("📈 IDF Values")
                st.markdown("*Inverse Document Frequency for each term*")
                st.dataframe(idf_df[important_terms], use_container_width=True)
                
                st.subheader("🔢 TF-IDF Matrix")
                st.markdown("*TF × IDF weights for each term in each document*")
                st.dataframe(tfidf_df[important_terms], use_container_width=True)
                
                if query_tfidf is not None and not query_tfidf.empty:
                    st.subheader("🔍 Query TF-IDF Vector")
                    st.markdown("*TF-IDF representation of your query*")
                    query_important = query_tfidf[important_terms]
                    # Show only non-zero values
                    non_zero_query = query_important.loc[:, (query_important != 0).any(axis=0)]
                    if not non_zero_query.empty:
                        st.dataframe(non_zero_query, use_container_width=True)
                    else:
                        st.info("Query terms not found in document vocabulary")
                
                if cos_df is not None:
                    st.subheader("🎯 Document Ranking (Cosine Similarity)")
                    st.markdown("*Documents ranked by relevance to your query*")
                    st.dataframe(cos_df, use_container_width=True)
                
                st.subheader("📏 Euclidean Normalized TF Matrix")
                st.markdown("*Length-normalized term frequencies*")
                st.dataframe(norm_tf_df[important_terms], use_container_width=True)
                
                # Show document content for reference
                with st.expander("📖 View Original Documents"):
                    docs_list = [d.strip() for d in docs_q2.split("\n") if d.strip()]
                    for i, doc in enumerate(docs_list):
                        st.write(f"**Doc{i+1}:** {doc}")
                
                # Show full vocabulary option
                with st.expander("🔍 View Complete Vocabulary"):
                    st.markdown("*All terms found in the documents*")
                    st.dataframe(tfidf_df, use_container_width=True)
                    
            else:
                st.error(f"❌ {result[1]}")

with tab3:
    st.header("Pointwise Mutual Information (PMI)")
    st.markdown("Analyze word associations using PMI scores.")
    
    docs_q3 = st.text_area("📝 Enter Documents (one per line)", height=150, key="docs_q3",
                          placeholder="The cat sat on the mat\nThe dog ran in the park\nCats and dogs are pets")
    target_word_q3 = st.text_input("🎯 Target Word", key="target_q3", placeholder="cat")
    
    if st.button("🚀 Run Analysis", key="run_q3"):
        with st.spinner("Computing PMI scores..."):
            pmi_df, target_sim = q3_pmi(docs_q3, target_word_q3)
            
            if pmi_df is not None:
                st.success("✅ Analysis completed!")
                
                st.subheader("📊 PMI Matrix")
                st.dataframe(pmi_df, use_container_width=True)
                
                if target_sim is not None:
                    st.subheader("🎯 PMI Scores for Target Word")
                    if isinstance(target_sim, pd.Series):
                        if len(target_sim) > 0:
                            target_df = pd.DataFrame({
                                'Word': target_sim.index,
                                'PMI Score': target_sim.values
                            })
                            st.dataframe(target_df, use_container_width=True)
                        else:
                            st.info("No significant associations found.")
                    else:
                        st.warning(str(target_sim))
            else:
                st.error(f"❌ {target_sim}")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### 📖 Instructions:
- **Q1**: Enter multiple documents (one per line) to compute cosine similarities between words
- **Q2**: Provide a CSV table with term frequencies and corresponding IDF values  
- **Q3**: Enter documents to compute PMI (Pointwise Mutual Information) between word pairs

**Example CSV format for Q2:**
```
term1,term2,term3
2,1,0
1,2,1
0,1,2
```
""")