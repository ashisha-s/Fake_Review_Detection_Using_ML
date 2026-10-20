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
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

# Hide only Streamlit's built-in top header / Deploy controls.
# All app widgets, headings, styling and prediction logic remain unchanged.
st.markdown("""
<style>
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"] {
    display: none !important;
}
.block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# NLTK SETUP
# ============================================================
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
# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================
@st.cache_resource
def load_model_files():
    model_folder = Path(__file__).resolve().parent / "models"
    model = joblib.load(
        model_folder / "logistic_regression_model.pkl"
    )
    tfidf = joblib.load(
        model_folder / "tfidf_vectorizer.pkl"
    )
    vader_scaler = joblib.load(
        model_folder / "vader_scaler.pkl"
    )
    return model, tfidf, vader_scaler
try:
    model, tfidf, vader_scaler = load_model_files()
except Exception as e:
    st.error("Unable to load the trained model files.")
    st.exception(e)
    st.stop()
# ============================================================
# VERIFY MODEL CONFIGURATION
# ============================================================
feature_names = tfidf.get_feature_names_out()
number_tfidf = len(feature_names)
if list(model.classes_) != [0, 1]:
    st.error(
        "Unexpected model labels. Expected 0 = Genuine "
        "and 1 = Fake."
    )
    st.stop()
if model.coef_.shape[1] != number_tfidf + 4:
    st.error(
        "Model feature count does not match "
        "TF-IDF + four VADER features."
    )
    st.stop()
# ============================================================
# TEXT PREPROCESSING
# ============================================================
def clean_text(text):
    text = str(text).lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    # Tokenization
    words = text.split()
    # Stop-word removal and lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)
# ============================================================
# VADER FEATURE EXTRACTION
# ============================================================
def extract_vader_features(text):
    scores = analyzer.polarity_scores(str(text))
    return np.array([
        scores["pos"],
        scores["neg"],
        scores["neu"],
        scores["compound"]
    ]).reshape(1, -1)
# ============================================================
# BUILD FEATURE VECTOR
# ============================================================
def build_feature_vector(review):
    # Stage 1: Preprocessing
    cleaned_review = clean_text(review)
    # Stage 2: TF-IDF
    tfidf_vector = tfidf.transform(
        [cleaned_review]
    )
    # Stage 3: VADER on original review
    vader_raw = extract_vader_features(review)
    # Scale VADER using saved training scaler
    vader_scaled = vader_scaler.transform(
        vader_raw
    )
    # Combine TF-IDF and VADER
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
# ============================================================
# TF-IDF INFORMATION
# ============================================================
def get_tfidf_information(tfidf_vector):
    row = tfidf_vector.getrow(0)
    information = []
    for index, value in zip(row.indices, row.data):
        information.append({
            "Term": feature_names[index],
            "TF-IDF Value": float(value)
        })
    # FIX: Define columns even when there are no active terms
    df = pd.DataFrame(
        information,
        columns=["Term", "TF-IDF Value"]
    )
    if not df.empty:
        df = df.sort_values(
            "TF-IDF Value",
            ascending=False
        )
    return df
# ============================================================
# LOGISTIC REGRESSION FEATURE CONTRIBUTIONS
# ============================================================
def calculate_feature_contributions(
    tfidf_vector,
    vader_scaled
):
    coefficients = model.coef_[0]
    # TF-IDF coefficients
    tfidf_coefficients = coefficients[:number_tfidf]
    row = tfidf_vector.getrow(0)
    word_information = []
    for index, value in zip(row.indices, row.data):
        coefficient = float(
            tfidf_coefficients[index]
        )
        contribution = float(
            value * coefficient
        )
        word_information.append({
            "Term": feature_names[index],
            "TF-IDF": float(value),
            "Coefficient": coefficient,
            "Contribution": contribution
        })
    # ========================================================
    # IMPORTANT FIX
    # ========================================================
    # Explicitly specify the columns.
    # This prevents KeyError: 'Contribution'
    # when the review has no active TF-IDF terms.
    word_df = pd.DataFrame(
        word_information,
        columns=[
            "Term",
            "TF-IDF",
            "Coefficient",
            "Contribution"
        ]
    )
    # VADER coefficients
    vader_coefficients = coefficients[
        number_tfidf:number_tfidf + 4
    ]
    vader_names = [
        "Positive",
        "Negative",
        "Neutral",
        "Compound"
    ]
    vader_information = []
    for i in range(4):
        scaled_value = float(
            vader_scaled[0][i]
        )
        coefficient = float(
            vader_coefficients[i]
        )
        contribution = scaled_value * coefficient
        vader_information.append({
            "Feature": vader_names[i],
            "Scaled Value": scaled_value,
            "Coefficient": coefficient,
            "Contribution": contribution
        })
    vader_df = pd.DataFrame(
        vader_information,
        columns=[
            "Feature",
            "Scaled Value",
            "Coefficient",
            "Contribution"
        ]
    )
    return word_df, vader_df
# ============================================================
# USER INTERFACE
# ============================================================
# ============================================================
# FIXED HEADER (same look, stays visible on scroll)
# ============================================================
st.markdown("""
<style>
.fixed-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background: var(--background-color, #ffffff);
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}
.fixed-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
    color: inherit;
}
</style>
<div class="fixed-header">
    <h1 padding="0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔍 Fake Review Detection Using Machine Learning</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height: 4.5rem;'></div>", unsafe_allow_html=True)
st.write(
    "Enter a hotel review to classify it as "
    "**Fake/Deceptive** or **Genuine/Truthful**."
)
st.caption(
    "Selected Model: Logistic Regression + TF-IDF + VADER"
)

# ============================================================
# REVIEW INPUT
# ============================================================
review = st.text_area(
    "Enter Hotel Review",
    height=200,
    placeholder="Type or paste a hotel review here..."
)
analyze = st.button(
    "Analyze Review",
    type="primary",
    use_container_width=True
)
# ============================================================
# ANALYSIS
# ============================================================
if analyze:
    if not review.strip():
        st.warning(
            "Please enter a review before analyzing."
        )
    else:
        try:
            (
                cleaned_review,
                tfidf_vector,
                vader_raw,
                vader_scaled,
                combined_features
            ) = build_feature_vector(review)
            # =================================================
            # MODEL PREDICTION
            # =================================================
            prediction = int(
                model.predict(combined_features)[0]
            )
            probabilities = model.predict_proba(
                combined_features
            )[0]
            probability_map = dict(
                zip(model.classes_, probabilities)
            )
            fake_probability = float(
                probability_map[1] * 100
            )
            genuine_probability = float(
                probability_map[0] * 100
            )
            # =================================================
            # PREDICTION RESULT
            # =================================================
            st.divider()
            st.header("Prediction Result")
            if prediction == 1:
                st.error(
                    "### 🚨 Fake / Deceptive Review"
                )
            else:
                st.success(
                    "### ✅ Genuine / Truthful Review"
                )
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Fake Probability",
                    f"{fake_probability:.2f}%"
                )
            with col2:
                st.metric(
                    "Genuine Probability",
                    f"{genuine_probability:.2f}%"
                )
            st.caption(
                "Label 1 = Fake/Deceptive | "
                "Label 0 = Genuine/Truthful"
            )
            # =================================================
            # MODEL EXPLANATION
            # =================================================
            st.divider()
            st.header(
                "How Did the Model Reach This Prediction?"
            )
            st.write(
                "The following stages explain how the "
                "review is processed and classified using "
                "the trained machine learning model."
            )
            # =================================================
            # STAGE 1: PREPROCESSING
            # =================================================
            with st.expander(
                "1️⃣ Text Preprocessing",
                expanded=True
            ):
                st.subheader("Original Review")
                st.info(review)
                st.subheader("Cleaned Review")
                if cleaned_review:
                    st.code(cleaned_review)
                else:
                    st.warning(
                        "The review contains no remaining "
                        "terms after preprocessing."
                    )
                st.write(
                    """
                    **Preprocessing operations:**
                    - Lowercase conversion
                    - HTML tag removal
                    - URL removal
                    - Number removal
                    - Punctuation removal
                    - Tokenization
                    - Stop-word removal
                    - Lemmatization
                    """
                )
            # =================================================
            # STAGE 2: TF-IDF
            # =================================================
            with st.expander(
                "2️⃣ TF-IDF Text Representation"
            ):
                st.write(
                    "TF-IDF transforms the cleaned review "
                    "into numerical feature values using "
                    "the vocabulary learned during training."
                )
                tfidf_df = get_tfidf_information(
                    tfidf_vector
                )
                st.metric(
                    "Active TF-IDF Features",
                    len(tfidf_df)
                )
                if not tfidf_df.empty:
                    st.subheader(
                        "Highest TF-IDF Terms"
                    )
                    top_tfidf = tfidf_df.head(15)
                    st.dataframe(
                        top_tfidf,
                        use_container_width=True,
                        hide_index=True
                    )
                    chart_data = top_tfidf[
                        ["Term", "TF-IDF Value"]
                    ].set_index("Term")
                    st.bar_chart(chart_data)
                else:
                    st.warning(
                        "No recognized TF-IDF vocabulary "
                        "terms were found in this review."
                    )
                st.caption(
                    "A high TF-IDF value does not by itself "
                    "mean a term indicates a fake review. "
                    "Its classification effect depends on "
                    "the Logistic Regression coefficient."
                )
            # =================================================
            # STAGE 3: VADER SENTIMENT
            # =================================================
            with st.expander(
                "3️⃣ VADER Sentiment Analysis"
            ):
                vader_names = [
                    "Positive",
                    "Negative",
                    "Neutral",
                    "Compound"
                ]
                vader_display = pd.DataFrame({
                    "Sentiment Feature": vader_names,
                    "Raw Score": vader_raw[0]
                })
                st.subheader(
                    "Extracted Sentiment Features"
                )
                st.dataframe(
                    vader_display,
                    use_container_width=True,
                    hide_index=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Positive",
                        f"{vader_raw[0][0]:.4f}"
                    )
                    st.metric(
                        "Negative",
                        f"{vader_raw[0][1]:.4f}"
                    )
                with col2:
                    st.metric(
                        "Neutral",
                        f"{vader_raw[0][2]:.4f}"
                    )
                    st.metric(
                        "Compound",
                        f"{vader_raw[0][3]:.4f}"
                    )
                st.caption(
                    "VADER scores are extracted from the "
                    "original review. The four scores are "
                    "then standardized using the saved "
                    "VADER scaler before classification."
                )
            # =================================================
            # CALCULATE FEATURE CONTRIBUTIONS
            # =================================================
            (
                word_contributions,
                vader_contributions
            ) = calculate_feature_contributions(
                tfidf_vector,
                vader_scaled
            )
            # =================================================
            # STAGE 4: FEATURE CONTRIBUTIONS
            # =================================================
            with st.expander(
                "4️⃣ Feature Contribution to Logistic Regression"
            ):
                st.write(
                    """
                    Logistic Regression uses a coefficient
                    for each input feature.
                    **Contribution = Feature Value × Coefficient**
                    - Positive contribution: pushes toward
                      **Fake/Deceptive (Label 1)**.
                    - Negative contribution: pushes toward
                      **Genuine/Truthful (Label 0)**.
                    These contributions explain the model's
                    numerical calculation, not the actual
                    truthfulness of the review.
                    """
                )
                st.subheader(
                    "TF-IDF Feature Contributions"
                )
                if not word_contributions.empty:
                    fake_terms = (
                        word_contributions[
                            word_contributions[
                                "Contribution"
                            ] > 0
                        ]
                        .sort_values(
                            "Contribution",
                            ascending=False
                        )
                        .head(10)
                    )
                    genuine_terms = (
                        word_contributions[
                            word_contributions[
                                "Contribution"
                            ] < 0
                        ]
                        .sort_values(
                            "Contribution",
                            ascending=True
                        )
                        .head(10)
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            "#### Terms Pushing Toward Fake"
                        )
                        if not fake_terms.empty:
                            st.dataframe(
                                fake_terms[
                                    ["Term", "Contribution"]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info(
                                "No active TF-IDF terms "
                                "pushed toward Fake."
                            )
                    with col2:
                        st.markdown(
                            "#### Terms Pushing Toward Genuine"
                        )
                        if not genuine_terms.empty:
                            st.dataframe(
                                genuine_terms[
                                    ["Term", "Contribution"]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info(
                                "No active TF-IDF terms "
                                "pushed toward Genuine."
                            )
                else:
                    st.info(
                        "No active TF-IDF terms were found. "
                        "The TF-IDF contribution is zero."
                    )
                # =============================================
                # VADER CONTRIBUTIONS
                # =============================================
                st.subheader(
                    "VADER Feature Contributions"
                )
                st.dataframe(
                    vader_contributions,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "VADER contributions are calculated "
                    "using standardized VADER values, "
                    "not raw sentiment scores."
                )
            # =================================================
            # STAGE 5: FINAL LOGISTIC REGRESSION DECISION
            # =================================================
            with st.expander(
                "5️⃣ Final Logistic Regression Decision",
                expanded=True
            ):
                decision_score = float(
                    model.decision_function(
                        combined_features
                    )[0]
                )
                intercept = float(
                    model.intercept_[0]
                )
                # FIX: Safe even if there are no TF-IDF terms
                tfidf_total = float(
                    word_contributions[
                        "Contribution"
                    ].sum()
                )
                vader_total = float(
                    vader_contributions[
                        "Contribution"
                    ].sum()
                )
                calculated_score = (
                    intercept +
                    tfidf_total +
                    vader_total
                )
                calculation_df = pd.DataFrame({
                    "Component": [
                        "Model Intercept",
                        "TF-IDF Contribution",
                        "VADER Contribution",
                        "Final Decision Score"
                    ],
                    "Value": [
                        intercept,
                        tfidf_total,
                        vader_total,
                        calculated_score
                    ]
                })
                st.subheader(
                    "Decision Score Calculation"
                )
                st.dataframe(
                    calculation_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.write(
                    "**Logistic Regression Formula:**"
                )
                st.latex(
                    r"z = b + \sum_i x_i w_i"
                )
                st.write(
                    "**Fake Probability Formula:**"
                )
                st.latex(
                    r"P(\mathrm{Fake}) = "
                    r"\frac{1}{1+e^{-z}}"
                )
                st.metric(
                    "Decision Score",
                    f"{decision_score:.4f}"
                )
                st.metric(
                    "Fake Probability",
                    f"{fake_probability:.2f}%"
                )
                st.metric(
                    "Genuine Probability",
                    f"{genuine_probability:.2f}%"
                )
                # Verify explanation matches model
                if np.isclose(
                    calculated_score,
                    decision_score,
                    atol=1e-6
                ):
                    st.success(
                        "Feature contributions match "
                        "the model's decision score."
                    )
                else:
                    st.warning(
                        "The calculated contributions "
                        "do not exactly match the model "
                        "decision score. Check feature order."
                    )
                if prediction == 1:
                    st.error(
                        "Final Decision → FAKE / DECEPTIVE"
                    )
                else:
                    st.success(
                        "Final Decision → GENUINE / TRUTHFUL"
                    )
        except Exception as e:
            st.error(
                "An error occurred while analyzing "
                "the review."
            )
            st.exception(e)
