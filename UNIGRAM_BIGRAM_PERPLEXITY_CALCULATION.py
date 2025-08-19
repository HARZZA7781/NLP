import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from nltk import word_tokenize, FreqDist, ngrams

st.set_page_config(page_title="N-gram Language Model", layout="wide")

if "mode" not in st.session_state:
    st.session_state.mode = "📊"

mode = st.session_state.mode

if mode.startswith("📊"):
    st.title("📘 N-gram Language Model Analyzer")
    uploaded_file = st.file_uploader("📁 Upload a plain text file", type=["txt"])

    if uploaded_file:
        try:
            corpus = uploaded_file.read().decode("utf-8").lower()
            st.subheader("📄 Corpus Preview")
            st.text_area("Corpus", corpus, height=200)

            tokens = [token for token in word_tokenize(corpus) if token.isalpha()]
            total_tokens = len(tokens)
            bigrams = list(ngrams(tokens, 2))

            # Frequency Distributions
            unigram_fd = FreqDist(tokens)
            bigram_fd = FreqDist(bigrams)
            prev_word_counts = FreqDist(w1 for (w1, _) in bigrams)

            # Probability Functions
            def unigram_prob(word):
                return unigram_fd[word] / total_tokens if total_tokens else 0

            def bigram_prob(w1, w2, laplace=False):
                if laplace:
                    vocab_size = len(unigram_fd)
                    return (bigram_fd[(w1, w2)] + 1) / (prev_word_counts[w1] + vocab_size)
                return bigram_fd[(w1, w2)] / prev_word_counts[w1] if prev_word_counts[w1] > 0 else 0

            st.subheader("🔍 Choose an Option")
            choice = st.radio("Select one:", [
                "1️⃣ Show all unigrams",
                "2️⃣ Lookup unigram probability of a word",
                "3️⃣ Show all bigrams",
                "4️⃣ Lookup bigram probability for two words",
                "5️⃣ Compute Perplexity for a sentence",
                "6️⃣ Visualize N-grams"
            ])

            if choice.startswith("1"):
                st.subheader("📊 Unigram Table")
                df_uni = pd.DataFrame({
                    "Word": list(unigram_fd.keys()),
                    "Count": list(unigram_fd.values()),
                    "Probability": [round(unigram_prob(w), 6) for w in unigram_fd]
                }).sort_values(by="Count", ascending=False)
                st.dataframe(df_uni)

            elif choice.startswith("2"):
                word = st.text_input("Enter a word:")
                if word:
                    count = unigram_fd[word]
                    prob = unigram_prob(word)
                    st.write(f"🔹 **Count:** {count}")
                    st.write(f"🔹 **Probability:** {round(prob, 6)}")

            elif choice.startswith("3"):
                st.subheader("📊 Bigram Table")
                bigram_data = []
                for (w1, w2), count in bigram_fd.items():
                    prob = bigram_prob(w1, w2)
                    bigram_data.append({"Bigram": f"{w1} {w2}", "Count": count, "Probability": round(prob, 6)})

                df_bi = pd.DataFrame(bigram_data).sort_values(by="Count", ascending=False)
                st.dataframe(df_bi)

            elif choice.startswith("4"):
                col1, col2 = st.columns(2)
                with col1:
                    w1 = st.text_input("First word:")
                with col2:
                    w2 = st.text_input("Second word:")

                if w1 and w2:
                    count = bigram_fd[(w1, w2)]
                    prob = bigram_prob(w1, w2)
                    st.write(f"🔹 **Bigram:** ({w1}, {w2})")
                    st.write(f"🔹 **Count:** {count}")
                    st.write(f"🔹 **Probability:** {round(prob, 6)}")

            elif choice.startswith("5"):
                st.subheader("📉 Perplexity Calculator")
                test_sentence = st.text_input("Enter a test sentence:", "I want English food")
                use_laplace = st.checkbox("Use Laplace Smoothing", value=True)

                if test_sentence:
                    test_tokens = [t for t in word_tokenize(test_sentence.lower()) if t.isalpha()]
                    test_bigrams = list(ngrams(test_tokens, 2))
                    N = len(test_bigrams)
                    log_prob_sum = 0
                    zero_prob = False

                    for w1, w2 in test_bigrams:
                        p = bigram_prob(w1, w2, laplace=use_laplace)
                        if p > 0:
                            log_prob_sum += math.log2(p)
                        else:
                            zero_prob = True
                            break

                    perplexity = 2 ** (-log_prob_sum / N) if not zero_prob else float('inf')
                    st.write(f"🔹 **Perplexity:** {perplexity:.4f}" if not zero_prob else "🔹 **Perplexity:** ∞ (Zero-probability bigram)")

            elif choice.startswith("6"):
                st.subheader("📊 Top 20 Unigrams: Frequency & Probability")
                top = unigram_fd.most_common(20)
                words, counts = zip(*top)
                probs = [unigram_prob(w) for w in words]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                sns.barplot(x=counts, y=words, ax=ax1)
                ax1.set_title("Top 20 Unigrams - Frequency")
                ax1.set_xlabel("Frequency")

                sns.barplot(x=probs, y=words, ax=ax2)
                ax2.set_title("Top 20 Unigrams - Probability")
                ax2.set_xlabel("Probability")

                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")





# ------------------ POS TAGGING ------------------
elif mode.startswith("🔤"):
    st.header("🔤 POS Tagging & HMM Analysis")

    sentence = st.text_input("Enter a sentence:", "I want to learn NLP")
    if sentence:
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        st.write("📌 **POS Tags:**", pos_tags)

        tag_freq = Counter(tag for _, tag in pos_tags)
        st.write("📊 **POS Tag Frequencies:**", dict(tag_freq))

        st.info("Using HMM-based model built from Brown corpus (news category)...")

        # Train HMM tagger on subset
        tagged_sents = brown.tagged_sents(categories='news')[:1000]
        cfd_trans = ConditionalFreqDist()
        cfd_emit = ConditionalFreqDist()

        for sent in tagged_sents:
            prev_tag = '<s>'
            for word, tag in sent:
                cfd_emit[tag][word.lower()] += 1
                cfd_trans[prev_tag][tag] += 1
                prev_tag = tag

        # Probability distributions with Lidstone smoothing
        transition_probs = {
            tag: LidstoneProbDist(cfd_trans[tag], 0.1, bins=len(cfd_trans[tag]))
            for tag in cfd_trans
        }
        emission_probs = {
            tag: LidstoneProbDist(cfd_emit[tag], 0.1, bins=len(cfd_emit[tag]))
            for tag in cfd_emit
        }

        st.success("✅ HMM tagger created using Brown corpus (first 1000 sentences)")

        st.subheader("🔍 Emission & Transition Probability Check")

        col1, col2 = st.columns(2)
        with col1:
            check_word = st.text_input("Word (for emission):", "food")
            check_tag = st.selectbox("Tag (for emission):", sorted(emission_probs.keys()))
            if check_tag:
                emission = emission_probs[check_tag].prob(check_word.lower())
                st.write(f"📌 **P({check_word} | {check_tag}) = {emission:.6f}**")

        with col2:
            if "<s>" in transition_probs:
                check_next_tag = st.selectbox("Tag after <s> (for transition):", sorted(transition_probs['<s>'].samples()))
                prob = transition_probs["<s>"].prob(check_next_tag)
                st.write(f"📌 **P({check_next_tag} | <s>) = {prob:.6f}**")




# ------------------ NER ------------------
elif mode.startswith("🧾"):
    st.header("🧾 Named Entity Recognition (IOB Format)")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("❌ spaCy model 'en_core_web_sm' not found. Run: `python -m spacy download en_core_web_sm`")
        st.stop()

    text = st.text_input("Enter a sentence:", "John lives in New York")

    if text:
        doc = nlp(text)
        ner_data = [
            (token.text, token.ent_iob_, token.ent_type_ if token.ent_type_ else "O")
            for token in doc
        ]

        ner_df = pd.DataFrame(ner_data, columns=["Token", "IOB Tag", "Entity Type"])
        st.subheader("📋 IOB Tagged Tokens")
        st.dataframe(ner_df)

        st.subheader("📌 Visual Highlight (Displacy)")
        html = spacy.displacy.render(doc, style="ent")
        components.html(html, height=300, scrolling=True)

        entity_freq = Counter(ent.label_ for ent in doc.ents)
        if entity_freq:
            st.write("📊 Named Entity Counts:", dict(entity_freq))
        else:
            st.info("ℹ️ No named entities found in the given sentence.")
