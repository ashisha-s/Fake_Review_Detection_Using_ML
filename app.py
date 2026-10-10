import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
import pandas as pd

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


# ============================================================
# NLTK SETUP
# ============================================================

@st.cache_resource
def setup_nltk():

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("vader_lexicon", quiet=True)

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

    model = joblib.load(
        "models/logistic_regression_model.pkl"
    )

    tfidf = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    vader_scaler = joblib.load(
        "models/vader_scaler.pkl"
    )

    return model, tfidf, vader_scaler


model, tfidf, vader_scaler = load_model_files()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# VADER
# ============================================================

def extract_vader_features(text):

    scores = analyzer.polarity_scores(
        str(text)
    )

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

    # Preprocessing
    cleaned_review = clean_text(review)

    # TF-IDF
    tfidf_vector = tfidf.transform(
        [cleaned_review]
    )

    # Raw VADER
    vader_raw = extract_vader_features(
        review
    )

    # Scale VADER exactly as during training
    vader_scaled = vader_scaler.transform(
        vader_raw
    )

    # Combine
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
# TF-IDF EXPLANATION
# ============================================================

def get_tfidf_information(tfidf_vector):

    feature_names = (
        tfidf.get_feature_names_out()
    )

    row = tfidf_vector.toarray()[0]

    active_indices = np.where(
        row > 0
    )[0]

    information = []

    for index in active_indices:

        information.append({
            "Term": feature_names[index],
            "TF-IDF Value": row[index]
        })

    df = pd.DataFrame(information)

    if not df.empty:

        df = df.sort_values(
            "TF-IDF Value",
            ascending=False
        )

    return df


# ============================================================
# LOGISTIC REGRESSION CONTRIBUTIONS
# ============================================================

def calculate_feature_contributions(
    tfidf_vector,
    vader_scaled
):

    coefficients = model.coef_[0]

    number_tfidf = len(
        tfidf.get_feature_names_out()
    )

    # --------------------------------------------------------
    # TF-IDF coefficients
    # --------------------------------------------------------

    tfidf_coefficients = (
        coefficients[:number_tfidf]
    )

    tfidf_values = (
        tfidf_vector.toarray()[0]
    )

    tfidf_contributions = (
        tfidf_values *
        tfidf_coefficients
    )

    feature_names = (
        tfidf.get_feature_names_out()
    )

    active_indices = np.where(
        tfidf_values > 0
    )[0]

    word_information = []

    for index in active_indices:

        contribution = (
            tfidf_contributions[index]
        )

        word_information.append({

            "Term":
                feature_names[index],

            "TF-IDF":
                tfidf_values[index],

            "Coefficient":
                tfidf_coefficients[index],

            "Contribution":
                contribution
        })


    word_df = pd.DataFrame(
        word_information
    )


    # --------------------------------------------------------
    # VADER coefficients
    # --------------------------------------------------------

    vader_coefficients = (
        coefficients[
            number_tfidf:
            number_tfidf + 4
        ]
    )

    vader_names = [
        "Positive",
        "Negative",
        "Neutral",
        "Compound"
    ]

    vader_information = []

    for i in range(4):

        contribution = (
            vader_scaled[0][i] *
            vader_coefficients[i]
        )

        vader_information.append({

            "Feature":
                vader_names[i],

            "Scaled Value":
                vader_scaled[0][i],

            "Coefficient":
                vader_coefficients[i],

            "Contribution":
                contribution
        })


    vader_df = pd.DataFrame(
        vader_information
    )


    return (
        word_df,
        vader_df
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🔍 Fake Review Detection Using Machine Learning"
)

st.write(
    "Enter a hotel review to classify it as "
    "**Fake/Deceptive** or **Genuine/Truthful**."
)

st.caption(
    "Selected Model: Logistic Regression + TF-IDF + VADER"
)

st.divider()


review = st.text_area(
    "Enter Hotel Review",
    height=200,
    placeholder=(
        "Type or paste a hotel review here..."
    )
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
            "Please enter a review."
        )

    else:

        (
            cleaned_review,
            tfidf_vector,
            vader_raw,
            vader_scaled,
            combined_features

        ) = build_feature_vector(review)


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            combined_features
        )[0]

        probabilities = model.predict_proba(
            combined_features
        )[0]

        probability_map = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        fake_probability = (
            probability_map[1] * 100
        )

        genuine_probability = (
            probability_map[0] * 100
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header("Prediction Result")


        if prediction == 1:

            st.error(
                "## 🚨 Fake / Deceptive"
            )

        else:

            st.success(
                "## ✅ Genuine / Truthful"
            )


        col1, col2 = st.columns(2)

        col1.metric(
            "Fake Probability",
            f"{fake_probability:.2f}%"
        )

        col2.metric(
            "Genuine Probability",
            f"{genuine_probability:.2f}%"
        )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.divider()

        st.header(
            "How Did the Model Reach This Prediction?"
        )

        st.write(
            "The following stages show how the entered "
            "review was transformed and processed by the "
            "trained machine-learning pipeline."
        )


        # ====================================================
        # STAGE 1 - PREPROCESSING
        # ====================================================

        with st.expander(
            "1️⃣ Text Preprocessing",
            expanded=True
        ):

            st.write(
                "**Original Review**"
            )

            st.info(review)


            st.write(
                "**Cleaned Review Used for TF-IDF**"
            )

            st.code(
                cleaned_review
            )


            st.write(
                """
                The preprocessing stage performs:

                - Lowercase conversion
                - HTML removal
                - URL removal
                - Number removal
                - Punctuation removal
                - Tokenization
                - Stop-word removal
                - Lemmatization
                """
            )


        # ====================================================
        # STAGE 2 - TF-IDF
        # ====================================================

        with st.expander(
            "2️⃣ TF-IDF Text Representation"
        ):

            tfidf_df = (
                get_tfidf_information(
                    tfidf_vector
                )
            )

            st.write(
                """
                TF-IDF converts the cleaned review into
                numerical values. Higher values indicate
                terms that receive greater weight in the
                TF-IDF representation.
                """
            )

            if not tfidf_df.empty:

                st.write(
                    "**Highest TF-IDF Terms**"
                )

                top_tfidf = (
                    tfidf_df.head(15)
                )

                st.dataframe(
                    top_tfidf,
                    use_container_width=True,
                    hide_index=True
                )


                chart_data = (
                    top_tfidf[
                        ["Term", "TF-IDF Value"]
                    ]
                    .set_index("Term")
                )

                st.bar_chart(
                    chart_data
                )

            else:

                st.warning(
                    "No terms from this review were "
                    "found in the fitted TF-IDF vocabulary."
                )


        # ====================================================
        # STAGE 3 - VADER
        # ====================================================

        with st.expander(
            "3️⃣ VADER Sentiment Analysis"
        ):

            vader_display = pd.DataFrame({

                "Sentiment Feature": [
                    "Positive",
                    "Negative",
                    "Neutral",
                    "Compound"
                ],

                "Raw Score": [
                    vader_raw[0][0],
                    vader_raw[0][1],
                    vader_raw[0][2],
                    vader_raw[0][3]
                ]
            })


            st.dataframe(
                vader_display,
                use_container_width=True,
                hide_index=True
            )


            st.write(
                f"""
                **Positive:** {vader_raw[0][0]:.4f}

                **Negative:** {vader_raw[0][1]:.4f}

                **Neutral:** {vader_raw[0][2]:.4f}

                **Compound:** {vader_raw[0][3]:.4f}
                """
            )


            st.caption(
                "These raw VADER values are standardized "
                "using the scaler fitted on the training "
                "data before being supplied to Logistic Regression."
            )


        # ====================================================
        # STAGE 4 - FEATURE CONTRIBUTION
        # ====================================================

        with st.expander(
            "4️⃣ Feature Contribution to Logistic Regression"
        ):

            (
                word_contributions,
                vader_contributions

            ) = calculate_feature_contributions(
                tfidf_vector,
                vader_scaled
            )


            st.write(
                """
                Logistic Regression assigns a coefficient
                to every input feature.

                For this review:

                **Contribution = Feature Value × Model Coefficient**

                A positive contribution pushes the decision
                toward **Fake/Deceptive (Label 1)**.

                A negative contribution pushes the decision
                toward **Genuine/Truthful (Label 0)**.
                """
            )


            # ================================================
            # WORD CONTRIBUTIONS
            # ================================================

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

                    st.subheader(
                        "Terms Pushing Toward Fake"
                    )

                    if not fake_terms.empty:

                        st.dataframe(
                            fake_terms[
                                [
                                    "Term",
                                    "Contribution"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.write(
                            "No active TF-IDF terms "
                            "pushed toward Fake."
                        )


                with col2:

                    st.subheader(
                        "Terms Pushing Toward Genuine"
                    )

                    if not genuine_terms.empty:

                        st.dataframe(
                            genuine_terms[
                                [
                                    "Term",
                                    "Contribution"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.write(
                            "No active TF-IDF terms "
                            "pushed toward Genuine."
                        )


            # ================================================
            # VADER CONTRIBUTIONS
            # ================================================

            st.subheader(
                "VADER Feature Contributions"
            )

            st.dataframe(
                vader_contributions,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # STAGE 5 - FINAL LOGISTIC REGRESSION CALCULATION
        # ====================================================

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


            st.write(
                "**Logistic Regression Calculation**"
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


            st.dataframe(
                calculation_df,
                use_container_width=True,
                hide_index=True
            )


            st.write(
                "**Decision Score:**",
                f"{decision_score:.4f}"
            )


            st.write(
                "**Fake Probability:**",
                f"{fake_probability:.2f}%"
            )


            st.write(
                "**Genuine Probability:**",
                f"{genuine_probability:.2f}%"
            )


            if decision_score >= 0:

                st.error(
                    "Final Decision → FAKE / DECEPTIVE"
                )

            else:

                st.success(
                    "Final Decision → GENUINE / TRUTHFUL"
                )


        # ====================================================
        # PIPELINE SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Pipeline"
        )

        st.code(
            """
Original Review
      │
      ├──────────────► VADER Sentiment
      │                       │
      │                       ▼
      │                 VADER Scaler
      │                       │
      ▼                       │
Text Preprocessing            │
      │                       │
      ▼                       │
    TF-IDF                    │
      │                       │
      └───────────┬───────────┘
                  ▼
       TF-IDF + VADER Features
                  │
                  ▼
        Logistic Regression
                  │
                  ▼
          Fake Probability
                  │
                  ▼
       Fake / Genuine Prediction
            """
        )


        st.caption(
            "The displayed contributions describe how the "
            "trained Logistic Regression model calculated "
            "its prediction for this review. They should "
            "not be interpreted as proof that a review is "
            "deceptive or truthful."
        )