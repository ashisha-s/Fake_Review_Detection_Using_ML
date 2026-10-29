import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard styling: compact layout to fit the main action & result on the first screen
st.markdown("""
<style>
/* Optimize container padding to fit the first screen */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1100px;
}

/* Hide only Streamlit Deploy button, keep sidebar collapse/expand controls fully functional */
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Ensure the sidebar toggle icon is always visible and prominent */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

/* Compact Header */
.app-header {
    margin-bottom: 0.8rem;
}
.app-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin: 0;
}

/* Compact Result Cards */
.result-box-fake {
    background-color: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    margin: 0.6rem 0;
}
.result-box-genuine {
    background-color: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    margin: 0.6rem 0;
}

/* Metric styling */
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.04);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# NLTK SETUP
@st.cache_resource
def setup_nltk():
    for resource in [
        "stopwords",
        "wordnet",
        "omw-1.4",
        "vader_lexicon"
    ]:
        nltk.download(resource, quiet=True)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    analyzer = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, analyzer

stop_words, lemmatizer, analyzer = setup_nltk()

  
# LOAD TRAINED MODEL FILES
 
@st.cache_resource
def load_model_files():
    model_folder = Path(__file__).resolve().parent / "models"
    model = joblib.load(model_folder / "logistic_regression_model.pkl")
    tfidf = joblib.load(model_folder / "tfidf_vectorizer.pkl")
    vader_scaler = joblib.load(model_folder / "vader_scaler.pkl")
    return model, tfidf, vader_scaler

try:
    model, tfidf, vader_scaler = load_model_files()
except Exception as e:
    st.error("Unable to load trained model files from the `models` folder.")
    st.exception(e)
    st.stop()

  
# VERIFY MODEL CONFIGURATION
  
feature_names = tfidf.get_feature_names_out()
number_tfidf = len(feature_names)

if list(model.classes_) != [0, 1]:
    st.error("Unexpected model classes. Expected 0 = Genuine and 1 = Fake.")
    st.stop()

if model.coef_.shape[1] != number_tfidf + 4:
    st.error("Feature dimension mismatch with TF-IDF + 4 VADER features.")
    st.stop()

  
# TEXT PREPROCESSING PIPELINE
  
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)

  
# VADER FEATURE EXTRACTION
  
def extract_vader_features(text):
    scores = analyzer.polarity_scores(str(text))
    return np.array([
        scores["pos"],
        scores["neg"],
        scores["neu"],
        scores["compound"]
    ]).reshape(1, -1)

  
# BUILD COMBINED FEATURE VECTOR
  
def build_feature_vector(review_text):
    cleaned_review = clean_text(review_text)
    tfidf_vector = tfidf.transform([cleaned_review])
    vader_raw = extract_vader_features(review_text)
    vader_scaled = vader_scaler.transform(vader_raw)
    combined_features = hstack([
        tfidf_vector,
        csr_matrix(vader_scaled)
    ]).tocsr()
    return (
        cleaned_review,
        tfidf_vector,
        vader_raw,
        vader_scaled,
        combined_features
    )

  
# TF-IDF INFORMATION EXTRACTOR
  
def get_tfidf_information(tfidf_vector):
    row = tfidf_vector.getrow(0)
    information = []
    for index, value in zip(row.indices, row.data):
        information.append({
            "Term": feature_names[index],
            "TF-IDF Value": float(value)
        })
    df = pd.DataFrame(information, columns=["Term", "TF-IDF Value"])
    if not df.empty:
        df = df.sort_values("TF-IDF Value", ascending=False)
    return df

  
# FEATURE CONTRIBUTIONS TO LOGISTIC REGRESSION
  
def calculate_feature_contributions(tfidf_vector, vader_scaled):
    coefficients = model.coef_[0]
    tfidf_coefficients = coefficients[:number_tfidf]
    row = tfidf_vector.getrow(0)
    word_information = []
    for index, value in zip(row.indices, row.data):
        coefficient = float(tfidf_coefficients[index])
        contribution = float(value * coefficient)
        word_information.append({
            "Term": feature_names[index],
            "TF-IDF": float(value),
            "Coefficient": coefficient,
            "Contribution": contribution
        })
    word_df = pd.DataFrame(
        word_information,
        columns=["Term", "TF-IDF", "Coefficient", "Contribution"]
    )

    vader_coefficients = coefficients[number_tfidf:number_tfidf + 4]
    vader_names = ["Positive", "Negative", "Neutral", "Compound"]
    vader_information = []
    for i in range(4):
        scaled_value = float(vader_scaled[0][i])
        coefficient = float(vader_coefficients[i])
        contribution = scaled_value * coefficient
        vader_information.append({
            "Feature": vader_names[i],
            "Standardized Value": scaled_value,
            "Coefficient": coefficient,
            "Contribution": contribution
        })
    vader_df = pd.DataFrame(
        vader_information,
        columns=["Feature", "Standardized Value", "Coefficient", "Contribution"]
    )
    return word_df, vader_df

  
# SAMPLE REVIEWS (From Academic Report Evaluation)
  
SAMPLE_REVIEWS = {
    "Deceptive Review 1 (Booking Issue)": (
        "We booked a room at the Hilton Chicago for two nights to stay the weekend but upon arrival our reservation could not be found. I even provided the hotel with our confirmation number but it still wasn't available! Fortunately another reservation had not shown up so there was space available for us. Once we settled into our room for the night (after a long flight), there was a bad odor coming from the bathroom. Apparently the cleaning solution they used was tainted and they had to re-clean it after our arrival. Needless to say, with both of these poor experiences in just one night, I will be hesitant to EVER stay in this hotel again, let alone any other Hilton around. Please consider someplace else before the Hilton Chicago!"
    ),
    "Genuine Review 1 (Boutique Stay)": (
        "Excellent location, feels like a boutique hotel. Right next to Neiman Marcus (V.expensive dept store) and 2 mins walk to John Hancock tower. As most American medium to high end hotels service was excellent from the bellhop to reception. Upgrade to a suite if you are staying longer than a couple of days as walking around the windy city can be tiring and a suite feels like a small apartment. The pub in the basement is good for a sandwich and a Beer. They will be refurbing so take adavantage of lower rates now as I feel they will be going for very high end clients after the remodelling."
    ),
    "Deceptive Review 2 (Room Complaints)": (
        "Extremely disappointed in our stay. This hotel did not look like what pictures portrayed online. I booked online thinking what a perfect place this was going to be, only to arrive and have a room without a view even though that is what was paid for. The room smelled musty and not clean. Garbage was not changed from prior person. Found small ants in bathroom near shower. Called to complain and someone came up 2 hours later and used bug spray. Overall this hotel is in a great location, but we would never visit again. Staff was not overly friendly or helpful and the room was not as all as it seemed online."
    ),
    "Genuine Review 2 (Anniversary Visit)": (
        "My husband & I stayed at the Fitzpatrick in early June 2004 for my birthday-great hotel! Location provides easy walking to Navy Pier, Marshall Field, Michigan Avenue & John Hancock. Room seemed spacious even though its only about 300 sq ft. Lots of room in bathroom; comfortable bed; very quiet, upscale hotel. We would definitely stay here again!"
    )
}

# Session state initialization for review input
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

  
# SIDEBAR
  
with st.sidebar:
    st.markdown("### Model Details")
    st.markdown("""
    - **Model:** Logistic Regression ($C=1.0$)
    - **Dataset:** Ott Corpus (1,600 reviews)
    - **Text Representation:** TF-IDF (10,000 features)
    - **Sentiment:** VADER Polarity Scores (Scaled)
    - **Test Accuracy:** 91.88%
    - **Test F1-Score:** 91.93%
    - **ROC-AUC:** 96.81%
    - **Unseen Dataset:** 500 reviews (452 correct)
    """)


  
# COMPACT HEADER (Fits cleanly at the top of the first page)
  
st.markdown("""
<div class="app-header">
    <h1 class="app-title">🔍 Fake Review Detection Using Machine Learning</h1>
    <p class="app-subtitle">
        Enter a hotel review below to classify it as <strong>Fake / Deceptive</strong> or <strong>Genuine / Truthful</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

  
# QUICK SAMPLE BUTTONS & TEXT INPUT
  
sample_cols = st.columns([1.3, 1.3, 1.3, 1.3, 0.8])
with sample_cols[0]:
    if st.button("Sample Fake 1", use_container_width=True):
        st.session_state["review_text"] = SAMPLE_REVIEWS["Deceptive Review 1 (Booking Issue)"]
        st.rerun()
with sample_cols[1]:
    if st.button("Sample Genuine 1", use_container_width=True):
        st.session_state["review_text"] = SAMPLE_REVIEWS["Genuine Review 1 (Boutique Stay)"]
        st.rerun()
with sample_cols[2]:
    if st.button("Sample Fake 2", use_container_width=True):
        st.session_state["review_text"] = SAMPLE_REVIEWS["Deceptive Review 2 (Room Complaints)"]
        st.rerun()
with sample_cols[3]:
    if st.button("Sample Genuine 2", use_container_width=True):
        st.session_state["review_text"] = SAMPLE_REVIEWS["Genuine Review 2 (Anniversary Visit)"]
        st.rerun()
with sample_cols[4]:
    if st.button("Clear", use_container_width=True):
        st.session_state["review_text"] = ""
        st.rerun()

# Review Input Text Area (Compact height = 105px to preserve vertical screen space)
review = st.text_area(
    "Hotel Review",
    value=st.session_state["review_text"],
    height=240,
    placeholder="Type or paste a hotel review here, or click one of the sample buttons above...",
    label_visibility="collapsed"
)

# Analyze Button (Full width, primary action)
analyze_btn = st.button("Analyze Review", type="primary", use_container_width=True)

  
# PREDICTION & RESULTS (Displayed directly on first screen)
  
if analyze_btn:
    if not review.strip():
        st.warning("Please enter a hotel review before clicking Analyze Review.")
    else:
        try:
            (
                cleaned_review,
                tfidf_vector,
                vader_raw,
                vader_scaled,
                combined_features
            ) = build_feature_vector(review)

            # Predictions
            prediction = int(model.predict(combined_features)[0])
            probabilities = model.predict_proba(combined_features)[0]
            prob_map = dict(zip(model.classes_, probabilities))
            fake_prob = float(prob_map[1] * 100)
            genuine_prob = float(prob_map[0] * 100)

            # Decision calculation
            decision_score = float(model.decision_function(combined_features)[0])
            intercept = float(model.intercept_[0])
            word_df, vader_df = calculate_feature_contributions(tfidf_vector, vader_scaled)

            tfidf_total = float(word_df["Contribution"].sum())
            vader_total = float(vader_df["Contribution"].sum())
            calculated_score = intercept + tfidf_total + vader_total

            # Compact Verdict Banner
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box-fake">
                    <strong style="color: #dc2626; font-size: 1.15rem;">🚨 Result: FAKE / DECEPTIVE REVIEW (Label 1)</strong>
                    <span style="float: right; font-weight: 700; color: #dc2626;">Confidence: {fake_prob:.2f}% Fake</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-genuine">
                    <strong style="color: #059669; font-size: 1.15rem;">✅ Result: GENUINE / TRUTHFUL REVIEW (Label 0)</strong>
                    <span style="float: right; font-weight: 700; color: #059669;">Confidence: {genuine_prob:.2f}% Genuine</span>
                </div>
                """, unsafe_allow_html=True)

            # Compact Metrics Row (Fits directly under verdict on the first page)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Fake Probability", f"{fake_prob:.2f}%")
            with m_col2:
                st.metric("Genuine Probability", f"{genuine_prob:.2f}%")
            with m_col3:
                st.metric("Decision Score (z)", f"{decision_score:+.4f}")
            with m_col4:
                confidence_rating = "High" if max(fake_prob, genuine_prob) >= 80 else ("Moderate" if max(fake_prob, genuine_prob) >= 60 else "Borderline")
                st.metric("Confidence Level", confidence_rating)

            # Compact Dual Progress Bar
            st.markdown(f"""
            <div style="background-color: rgba(128,128,128,0.2); border-radius: 9999px; height: 12px; overflow: hidden; display: flex; margin: 6px 0 2px 0;">
                <div style="width: {genuine_prob}%; background-color: #10b981;" title="Genuine: {genuine_prob:.2f}%"></div>
                <div style="width: {fake_prob}%; background-color: #ef4444;" title="Fake: {fake_prob:.2f}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.8rem;">
                <span style="color: #059669;">● Genuine Share: {genuine_prob:.2f}%</span>
                <span style="color: #dc2626;">● Fake Share: {fake_prob:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Prompt to scroll down for technical breakdown
            st.caption("👇 Scroll down to see how the model processes text, extracts VADER sentiment, and calculates the score.")

            # =========================================================
            # DETAILED WORKING SECTION (Visible after scrolling down)
            # =========================================================
            st.subheader("How the Prediction Works")
            st.write("Detailed breakdown of text cleaning, TF-IDF representations, VADER sentiment features, and score calculations:")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "1. Text Preprocessing",
                "2. TF-IDF Features",
                "3. VADER Sentiment",
                "4. Feature Contributions",
                "5. Decision Score Calculation"
            ])

            # ---------------------------------------------------------
            # TAB 1: TEXT PREPROCESSING
            # ---------------------------------------------------------
            with tab1:
                st.markdown("#### Text Cleaning & Preprocessing")
                st.write(
                    "The review text is cleaned through lowercase conversion, removal of HTML tags, URLs, numbers, "
                    "punctuation, tokenization, English stop-word removal, and lemmatization."
                )

                prep_col1, prep_col2 = st.columns(2)
                with prep_col1:
                    st.caption("Original Review Text")
                    st.info(review)
                    st.caption(f"Word count: {len(review.split())}")
                with prep_col2:
                    st.caption("Cleaned Review Text")
                    if cleaned_review:
                        st.code(cleaned_review, language="text")
                        st.caption(f"Token count: {len(cleaned_review.split())}")
                    else:
                        st.warning("No tokens remaining after stop-word filtering.")

            # ---------------------------------------------------------
            # TAB 2: TF-IDF ANALYSIS
            # ---------------------------------------------------------
            with tab2:
                st.markdown("#### TF-IDF Feature Representation")
                st.write(
                    "The cleaned review is transformed using the fitted TF-IDF vectorizer (unigrams and bigrams, max 10,000 features)."
                )

                tfidf_df = get_tfidf_information(tfidf_vector)
                st.metric("Active Recognized Vocabulary Terms", len(tfidf_df))

                if not tfidf_df.empty:
                    top_tfidf = tfidf_df.head(15)
                    st.markdown("##### Highest TF-IDF Terms in this Review")
                    st.bar_chart(top_tfidf.set_index("Term"))
                    st.dataframe(top_tfidf, use_container_width=True, hide_index=True)
                else:
                    st.warning("No recognized TF-IDF vocabulary terms found in this review.")

            # ---------------------------------------------------------
            # TAB 3: VADER SENTIMENT
            # ---------------------------------------------------------
            with tab3:
                st.markdown("#### VADER Sentiment Analysis")
                st.write(
                    "VADER analyzes sentiment from the original review text, producing four polarity scores: Positive, Negative, Neutral, and Compound."
                )

                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                with v_col1:
                    st.metric("Positive", f"{vader_raw[0][0]:.4f}")
                with v_col2:
                    st.metric("Negative", f"{vader_raw[0][1]:.4f}")
                with v_col3:
                    st.metric("Neutral", f"{vader_raw[0][2]:.4f}")
                with v_col4:
                    st.metric("Compound", f"{vader_raw[0][3]:.4f}")

                compound_val = vader_raw[0][3]
                if compound_val >= 0.05:
                    st.success(f"Sentiment Tone: **Positive** (Compound: {compound_val:.4f})")
                elif compound_val <= -0.05:
                    st.error(f"Sentiment Tone: **Negative** (Compound: {compound_val:.4f})")
                else:
                    st.info(f"Sentiment Tone: **Neutral** (Compound: {compound_val:.4f})")

                st.markdown("##### Standardized VADER Features (Supplied to Model)")
                vader_display_df = pd.DataFrame({
                    "Sentiment Feature": ["Positive", "Negative", "Neutral", "Compound"],
                    "Raw Score": vader_raw[0],
                    "Standardized Value (Scaler)": vader_scaled[0]
                })
                st.dataframe(vader_display_df, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------
            # TAB 4: FEATURE CONTRIBUTIONS
            # ---------------------------------------------------------
            with tab4:
                st.markdown("#### Feature Influence on the Prediction")
                st.write(
                    "Contribution of each feature is calculated as: **Feature Value × Model Coefficient**."
                )
                st.markdown("""
                - **Positive Contribution (> 0):** Pushes prediction toward **Fake / Deceptive (Label 1)**.
                - **Negative Contribution (< 0):** Pushes prediction toward **Genuine / Truthful (Label 0)**.
                """)

                if not word_df.empty:
                    fake_push = word_df[word_df["Contribution"] > 0].sort_values("Contribution", ascending=False).head(10)
                    genuine_push = word_df[word_df["Contribution"] < 0].sort_values("Contribution", ascending=True).head(10)

                    fc_col1, fc_col2 = st.columns(2)
                    with fc_col1:
                        st.markdown("##### Terms Pushing Toward Fake")
                        if not fake_push.empty:
                            st.dataframe(
                                fake_push[["Term", "TF-IDF", "Coefficient", "Contribution"]],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("No active terms pushed toward Fake.")

                    with fc_col2:
                        st.markdown("##### Terms Pushing Toward Genuine")
                        if not genuine_push.empty:
                            st.dataframe(
                                genuine_push[["Term", "TF-IDF", "Coefficient", "Contribution"]],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("No active terms pushed toward Genuine.")
                else:
                    st.info("No active TF-IDF terms in this review.")

                st.markdown("##### VADER Feature Directional Contributions")
                st.dataframe(vader_df, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------
            # TAB 5: DECISION SCORE CALCULATION
            # ---------------------------------------------------------
            with tab5:
                st.markdown("#### Logistic Regression Calculation")
                st.write("The linear decision score $z$ is calculated and converted into probability via the sigmoid function:")

                st.latex(r"z = b + \sum_{i} x_i w_i")
                st.latex(r"P(\text{Fake}) = \frac{1}{1 + e^{-z}} \quad \text{and} \quad P(\text{Genuine}) = 1 - P(\text{Fake})")

                calc_summary_df = pd.DataFrame({
                    "Component": [
                        "Model Intercept (b)",
                        "Total TF-IDF Contribution",
                        "Total VADER Contribution",
                        "Calculated Decision Score (z)"
                    ],
                    "Value": [
                        f"{intercept:+.5f}",
                        f"{tfidf_total:+.5f}",
                        f"{vader_total:+.5f}",
                        f"{calculated_score:+.5f}"
                    ]
                })
                st.dataframe(calc_summary_df, use_container_width=True, hide_index=True)

                if np.isclose(calculated_score, decision_score, atol=1e-5):
                    st.success("Verification: Calculated contribution sum matches model decision function.")
                else:
                    st.info(f"Model Decision Value: {decision_score:.5f}")

                st.info(f"Decision Rule: If $z \\ge 0$ (or $P(\\text{{Fake}}) \\ge 50\\%$), classify as **Fake**; otherwise classify as **Genuine**.")

        except Exception as e:
            st.error("An error occurred while analyzing the review.")
            st.exception(e)