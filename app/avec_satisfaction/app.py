"""
Telco Customer Churn
====================
Application Streamlit de prédiction du churn
avec un modèle de Gradient Boosting.
"""

from pathlib import Path
import pickle
import joblib
import pandas as pd
import streamlit as st


# ================================================================
# CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ================================================================
# CHEMINS DES FICHIERS
# ================================================================

# Dossier contenant app.py
APP_DIR = Path(__file__).resolve().parent

# Racine du projet Churn_client/
PROJECT_DIR = APP_DIR.parent.parent

# Dossier contenant les modèles avec SatisfactionScore
MODEL_DIR = PROJECT_DIR / "models" / "avec_satisfaction"


MODEL_FILES = {
    "model": MODEL_DIR / "gradient_boosting_churn.pkl",
    "scaler": MODEL_DIR / "scaler.pkl",
    "frequency_encodings": MODEL_DIR / "frequency_encodings.pkl",
    "model_information": MODEL_DIR / "model_information.pkl"
}


YES_NO = ["Yes", "No"]
NO_YES = ["No", "Yes"]


# ================================================================
# STYLE
# ================================================================

st.markdown(
    """
    <style>

    /* ============================================================
       PAGE
       ============================================================ */

    .stApp {
        background-color: #f7f7f7;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* ============================================================
       TITRES
       ============================================================ */

    h1, h2, h3 {
        color: #222222 !important;
    }

    h1 {
        font-size: 30px !important;
        font-weight: 750 !important;
    }

    h2 {
        font-size: 24px !important;
    }

    h3 {
        font-size: 20px !important;
    }


    /* ============================================================
       EN-TETE
       ============================================================ */

    .main-title {
        margin-bottom: 4px;
    }

    .main-description {
        color: #666666;
        font-size: 15px;
        margin-top: 0px;
        margin-bottom: 18px;
    }

    .orange-line {
        height: 4px;
        width: 70px;
        background-color: #ff7900;
        border-radius: 10px;
        margin-bottom: 25px;
    }


    /* ============================================================
       ONGLETS
       ============================================================ */

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #eeeeee;
        padding: 5px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff7900 !important;
        color: white !important;
    }


    /* ============================================================
       BOUTONS
       ============================================================ */

    .stButton > button {
        background-color: #ff7900 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        height: 50px !important;
    }

    .stButton > button:hover {
        background-color: #e66d00 !important;
    }


    /* ============================================================
       INPUTS
       ============================================================ */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }


    /* ============================================================
       METRICS
       ============================================================ */

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 20px;
    }

    div[data-testid="stMetricValue"] {
        color: #ff7900 !important;
        font-weight: 800;
    }


    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {
        background-color: #202020;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }


    /* ============================================================
       SEPARATEUR
       ============================================================ */

    hr {
        border-color: #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# CHARGEMENT DU MODELE
# ================================================================

@st.cache_resource
def load_model():

    # Vérification de l'existence des fichiers
    missing_files = [
        str(path)
        for path in MODEL_FILES.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Les fichiers suivants sont introuvables :\n"
            + "\n".join(missing_files)
        )

    # ------------------------------------------------------------
    # Modèle
    # ------------------------------------------------------------

    model = joblib.load(
        MODEL_FILES["model"]
    )

    # ------------------------------------------------------------
    # Scaler
    # ------------------------------------------------------------

    scaler = joblib.load(
        MODEL_FILES["scaler"]
    )

    # ------------------------------------------------------------
    # Frequency Encoding
    # ------------------------------------------------------------

    with open(
        MODEL_FILES["frequency_encodings"],
        "rb"
    ) as f:

        frequency_encodings = pickle.load(f)

    # ------------------------------------------------------------
    # Informations du modèle
    # ------------------------------------------------------------

    with open(
        MODEL_FILES["model_information"],
        "rb"
    ) as f:

        model_information = pickle.load(f)

    return (
        model,
        scaler,
        frequency_encodings,
        model_information
    )


# ================================================================
# CHARGEMENT
# ================================================================

try:

    (
        model,
        scaler,
        frequency_encodings,
        model_information
    ) = load_model()

except Exception as e:

    st.error(
        "Impossible de charger le modèle."
    )

    st.write(
        "Vérifiez que les quatre fichiers du modèle "
        "sont présents dans le dossier :"
    )

    st.code(
        str(MODEL_DIR)
    )

    st.write(
        "Fichiers recherchés :"
    )

    st.code(
        "\n".join(
            str(path)
            for path in MODEL_FILES.values()
        )
    )

    st.exception(e)

    st.stop()


# ================================================================
# INFORMATIONS DU MODELE
# ================================================================

numeric_features = list(
    model_information["numeric_features"]
)

categorical_features = list(
    model_information["categorical_features"]
)

model_columns = list(
    model_information["columns"]
)


if not model_columns:

    st.error(
        "Aucune colonne du modèle n'a été trouvée."
    )

    st.stop()


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.title("📊 Telco Churn")

    st.divider()

    st.subheader("Modèle")

    st.info(
        "Gradient Boosting"
    )

    st.subheader("Variables")

    st.info(
        f"{len(model_columns)} variables utilisées"
    )

    st.subheader("Objectif")

    st.write(
        "Prédire le risque de départ "
        "d'un client."
    )

    st.divider()

    st.caption(
        "Telco Customer Churn"
    )


# ================================================================
# EN-TETE PRINCIPAL
# ================================================================

st.markdown(
    '<div class="main-title">',
    unsafe_allow_html=True
)

st.title(
    "Analyse du churn client"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Analysez le profil d'un client et estimez "
    "son risque de résiliation."
)

st.markdown(
    '<div class="orange-line"></div>',
    unsafe_allow_html=True
)


# ================================================================
# FORMULAIRE
# ================================================================

tab_profil, tab_services, tab_contrat, tab_usage = st.tabs(
    [
        "👤 Profil",
        "📡 Services",
        "💳 Contrat & facturation",
        "📊 Usage & valeur"
    ]
)


# ================================================================
# PROFIL
# ================================================================

with tab_profil:

    st.subheader(
        "Informations générales"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=46,
            step=1
        )

        gender = st.selectbox(
            "Genre",
            ["Male", "Female"]
        )

        under30 = st.selectbox(
            "Moins de 30 ans",
            NO_YES
        )

        senior = st.selectbox(
            "Senior Citizen",
            NO_YES
        )

    with col2:

        married = st.selectbox(
            "Marié(e)",
            NO_YES
        )

        dependents = st.selectbox(
            "Personnes à charge",
            NO_YES
        )

        number_dependents = st.number_input(
            "Nombre de personnes à charge",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        city = st.text_input(
            "Ville",
            value="Los Angeles"
        )

    with col3:

        referred_friend = st.selectbox(
            "Recommandé par un ami",
            NO_YES
        )

        number_referrals = st.number_input(
            "Nombre de recommandations",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        tenure = st.number_input(
            "Ancienneté (mois)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )


# ================================================================
# SERVICES
# ================================================================

with tab_services:

    st.subheader(
        "Services souscrits"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        phone_service = st.selectbox(
            "Service téléphonique",
            YES_NO
        )

        multiple_lines = st.selectbox(
            "Lignes multiples",
            YES_NO
        )

        internet_service = st.selectbox(
            "Service Internet",
            YES_NO
        )

        internet_type = st.selectbox(
            "Type Internet",
            [
                "Fiber Optic",
                "DSL",
                "Cable",
                "No Internet"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Sécurité en ligne",
            YES_NO
        )

        online_backup = st.selectbox(
            "Sauvegarde en ligne",
            YES_NO
        )

        device_protection = st.selectbox(
            "Protection appareil",
            YES_NO
        )

        tech_support = st.selectbox(
            "Support technique premium",
            YES_NO
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            YES_NO
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            YES_NO
        )

        streaming_music = st.selectbox(
            "Streaming Music",
            YES_NO
        )

        unlimited_data = st.selectbox(
            "Données illimitées",
            YES_NO
        )


# ================================================================
# CONTRAT ET FACTURATION
# ================================================================

with tab_contrat:

    st.subheader(
        "Offre, contrat et paiement"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        offer = st.selectbox(
            "Offre",
            [
                "No Offer",
                "Offer A",
                "Offer B",
                "Offer C",
                "Offer D",
                "Offer E"
            ]
        )

        contract = st.selectbox(
            "Type de contrat",
            [
                "Month-to-Month",
                "One Year",
                "Two Year"
            ]
        )

    with col2:

        paperless = st.selectbox(
            "Facturation électronique",
            YES_NO
        )

        payment = st.selectbox(
            "Méthode de paiement",
            [
                "Bank Withdrawal",
                "Credit Card",
                "Mailed Check"
            ]
        )

    with col3:

        monthly_charge = st.number_input(
            "Facture mensuelle",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total des charges",
            min_value=0.0,
            max_value=20000.0,
            value=1500.0,
            step=10.0
        )


# ================================================================
# USAGE ET VALEUR
# ================================================================

with tab_usage:

    st.subheader(
        "Usage et valeur client"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        avg_long_distance = st.number_input(
            "Frais moyens appels longue distance",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

        avg_gb = st.number_input(
            "Téléchargement moyen GB/mois",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

        total_long_distance = st.number_input(
            "Total appels longue distance",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=10.0
        )

    with col2:

        total_refunds = st.number_input(
            "Total des remboursements",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0
        )

        total_extra_data = st.number_input(
            "Total frais données supplémentaires",
            min_value=0.0,
            max_value=1000.0,
            value=5.0,
            step=1.0
        )

        total_revenue = st.number_input(
            "Revenu total",
            min_value=0.0,
            max_value=30000.0,
            value=2000.0,
            step=10.0
        )

    with col3:

        satisfaction = st.slider(
            "Score de satisfaction",
            min_value=1,
            max_value=5,
            value=3
        )

        cltv = st.number_input(
            "CLTV",
            min_value=1000,
            max_value=10000,
            value=4500,
            step=100
        )


# ================================================================
# DONNEES CLIENT
# ================================================================

client_data = {

    "Gender": gender,
    "Age": age,
    "Under30": under30,
    "SeniorCitizen": senior,
    "Married": married,
    "Dependents": dependents,
    "NumberofDependents": number_dependents,
    "City": city,
    "ReferredaFriend": referred_friend,
    "Number_of_Referrals": number_referrals,
    "TenureinMonths": tenure,
    "Offer": offer,

    "PhoneService": phone_service,
    "AvgMonthlyLongDistanceCharges": avg_long_distance,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "InternetType": internet_type,
    "AvgMonthlyGBDownload": avg_gb,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtectionPlan": device_protection,
    "PremiumTechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "StreamingMusic": streaming_music,
    "UnlimitedData": unlimited_data,

    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,

    "MonthlyCharge": monthly_charge,
    "TotalCharges": total_charges,
    "TotalRefunds": total_refunds,
    "TotalExtraDataCharges": total_extra_data,
    "TotalLongDistanceCharges": total_long_distance,
    "TotalRevenue": total_revenue,
    "SatisfactionScore": satisfaction,
    "CLTV": cltv
}


input_df = pd.DataFrame(
    [client_data]
)


# ================================================================
# PREPARATION DES DONNEES
# ================================================================

def check_columns(data):

    missing = [
        col
        for col in model_columns
        if col not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )


def prepare_data(data):

    data = data.copy()

    check_columns(data)

    # ------------------------------------------------------------
    # Frequency Encoding
    # ------------------------------------------------------------

    for col in categorical_features:

        if col not in frequency_encodings:

            raise ValueError(
                f"Encodage manquant pour : {col}"
            )

        data[col] = (
            data[col]
            .map(frequency_encodings[col])
            .fillna(0)
        )

    # ------------------------------------------------------------
    # Vérification des variables numériques
    # ------------------------------------------------------------

    missing_numeric = [
        col
        for col in numeric_features
        if col not in data.columns
    ]

    if missing_numeric:

        raise ValueError(
            "Variables numériques manquantes : "
            f"{missing_numeric}"
        )

    # ------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------

    data[numeric_features] = scaler.transform(
        data[numeric_features]
    )

    return data[model_columns]


# ================================================================
# ANALYSE
# ================================================================

st.divider()

st.subheader(
    "Analyse"
)

st.write(
    "Lorsque toutes les informations sont renseignées, "
    "lancez l'analyse pour obtenir la prédiction."
)

run_prediction = st.button(
    "🟠 Analyser le risque de churn",
    type="primary",
    use_container_width=True
)


# ================================================================
# RESULTAT
# ================================================================

if run_prediction:

    try:

        # --------------------------------------------------------
        # Préparation
        # --------------------------------------------------------

        prepared_data = prepare_data(
            input_df
        )

        # --------------------------------------------------------
        # Prédiction
        # --------------------------------------------------------

        prediction = model.predict(
            prepared_data
        )[0]

        probability = model.predict_proba(
            prepared_data
        )[0][1]

        churn_probability = probability * 100

        st.divider()

        st.subheader(
            "Résultat de l'analyse"
        )

        result_col1, result_col2 = st.columns(2)

        # --------------------------------------------------------
        # STATUT
        # --------------------------------------------------------

        with result_col1:

            if prediction == 1:

                st.error(
                    "🔴 Risque de churn"
                )

                st.write(
                    "Le modèle estime que ce client "
                    "présente un risque de quitter "
                    "l'entreprise."
                )

            else:

                st.success(
                    "🟢 Pas de churn"
                )

                st.write(
                    "Le modèle estime que ce client "
                    "présente un faible risque de "
                    "quitter l'entreprise."
                )

        # --------------------------------------------------------
        # PROBABILITE
        # --------------------------------------------------------

        with result_col2:

            st.metric(
                "Probabilité de churn",
                f"{churn_probability:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # --------------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------------

        st.write("")

        if churn_probability >= 70:

            st.warning(
                "⚠️ Risque élevé : la probabilité "
                "de churn est élevée. Une action de "
                "rétention peut être envisagée."
            )

        elif churn_probability >= 40:

            st.info(
                "ℹ️ Risque modéré : le client présente "
                "un risque intermédiaire. Une surveillance "
                "peut être pertinente."
            )

        else:

            st.success(
                "✅ Risque faible : le client semble "
                "relativement stable selon le modèle."
            )

        # --------------------------------------------------------
        # DONNEES TRANSMISES
        # --------------------------------------------------------

        with st.expander(
            "🔎 Voir les données transmises au modèle"
        ):

            st.dataframe(
                prepared_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "Une erreur est survenue lors de la prédiction."
        )

        st.exception(e)


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Telco Customer Churn Prediction • "
    "Gradient Boosting • Machine Learning"
)
