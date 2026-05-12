import os
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    silhouette_score
)
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    PageBreak,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")

start = time.time()

BASE_DIR = os.getcwd()
INPUT_BASE = BASE_DIR

OUT = os.path.join(BASE_DIR, "Output")
CHART = os.path.join(OUT, "charts")
UNZIP_DIR = os.path.join(BASE_DIR, "unzipped")

os.makedirs(OUT, exist_ok=True)
os.makedirs(CHART, exist_ok=True)
os.makedirs(UNZIP_DIR, exist_ok=True)

report_path = os.path.join(
    OUT,
    "NATO_Complete_Analysis_Report.pdf"
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "title",
    fontSize=24,
    alignment=TA_CENTER,
    textColor=HexColor("#0f172a"),
    spaceAfter=12
)

subtitle_style = ParagraphStyle(
    "subtitle",
    fontSize=14,
    alignment=TA_CENTER,
    textColor=HexColor("#2563eb"),
    spaceAfter=25
)

heading_style = ParagraphStyle(
    "heading",
    fontSize=18,
    textColor=HexColor("#1d4ed8"),
    spaceAfter=14
)

body_style = ParagraphStyle(
    "body",
    fontSize=11,
    leading=16,
    spaceAfter=10
)

doc = SimpleDocTemplate(
    report_path,
    pagesize=A4,
    leftMargin=40,
    rightMargin=40,
    topMargin=40,
    bottomMargin=40
)

elements = []

elements.append(
    Paragraph(
        "NATO Alliance Complete Dataset 2024",
        title_style
    )
)

elements.append(
    Paragraph(
        "Advanced Data Science, Time-Series and Machine Learning Analysis",
        subtitle_style
    )
)

intro = """
This report presents statistically meaningful exploratory data analysis,
machine learning, clustering, anomaly detection, NATO-specific insights,
and temporal trend analysis for NATO datasets.
"""

elements.append(
    Paragraph(intro, body_style)
)

elements.append(PageBreak())


def collect_csv_files():

    csv_files = []

    for root, dirs, files in os.walk(INPUT_BASE):

        for file in files:

            full_path = os.path.join(
                root,
                file
            )

            if file.lower().endswith(".csv"):
                csv_files.append(full_path)

            elif file.lower().endswith(".zip"):

                try:
                    with zipfile.ZipFile(
                        full_path,
                        "r"
                    ) as z:

                        z.extractall(
                            UNZIP_DIR
                        )

                except:
                    continue

    for root, dirs, files in os.walk(
        UNZIP_DIR
    ):
        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, file)
                )

    return list(set(csv_files))


csv_files = collect_csv_files()

if len(csv_files) == 0:
    raise ValueError(
        f"No CSV files found in: {INPUT_BASE}"
    )


def clean_columns(df):

    df.columns = [
        c.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        for c in df.columns
    ]

    return df


def remove_leakage_columns(df):

    bad_cols = [
        "record_id",
        "id",
        "serial_no",
        "index"
    ]

    existing = [
        c for c in bad_cols
        if c in df.columns
    ]

    df = df.drop(
        columns=existing,
        errors="ignore"
    )

    return df


def choose_target(df):

    priority_targets = [
        "defense_budget_billion_usd",
        "mission_cost_m_usd",
        "combat_ready_pct",
        "troops_deployed",
        "units_count",
        "gdp_billion_usd",
        "total_value_m_usd"
    ]

    for col in priority_targets:

        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_cols) > 0:
        return numeric_cols[0]

    return None


def preprocess_data(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    processed = preprocessor.fit_transform(df)

    return (
        processed,
        numeric_cols,
        categorical_cols
    )


all_summary = []
for idx, file_path in enumerate(csv_files):

    dataset_name = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path)
    except:
        continue

    df = clean_columns(df)
    df = remove_leakage_columns(df)

    if len(df) == 0:
        continue

    target_col = choose_target(df)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns.tolist()

    if len(numeric_cols) == 0:
        continue

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[numeric_cols] = (
        df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
    )

    df[numeric_cols] = (
        df[numeric_cols]
        .fillna(
            df[numeric_cols]
            .median()
        )
    )

    charts = []
    descriptions = []

    section_title = (
        f"Dataset Analysis: {dataset_name}"
    )

    elements.append(
        Paragraph(
            section_title,
            heading_style
        )
    )

    # -----------------------------
    # NUMERIC CORRELATION ONLY
    # -----------------------------

    usable_numeric = [
        c for c in numeric_cols
        if df[c].nunique() > 1
    ]

    if len(usable_numeric) > 1:

        fig = plt.figure(
            figsize=(12, 9)
        )

        corr = (
            df[usable_numeric]
            .corr()
        )

        plt.imshow(
            corr,
            aspect="auto"
        )

        plt.colorbar()

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns
        )

        plt.title(
            f"Correlation Matrix - {dataset_name}"
        )

        corr_path = os.path.join(
            CHART,
            f"corr_{idx}.png"
        )

        fig.savefig(
            corr_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(corr_path)

        descriptions.append(
            "Correlation matrix of valid numerical variables only."
        )

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------

    limited_numeric = [
        c for c in usable_numeric
        if c != target_col
    ][:6]

    for col in limited_numeric:

        fig = plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            df[col],
            bins=30
        )

        plt.title(
            f"{col} Distribution"
        )

        hist_path = os.path.join(
            CHART,
            f"{idx}_{col}_hist.png"
        )

        fig.savefig(
            hist_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(hist_path)

        descriptions.append(
            f"Distribution of {col}."
        )

    # -----------------------------
    # TARGET RELATIONSHIPS
    # -----------------------------

    if target_col in df.columns:

        strongest_features = []

        try:
            correlations = (
                df[usable_numeric]
                .corr()[target_col]
                .abs()
                .sort_values(
                    ascending=False
                )
            )

            strongest_features = [
                c for c in correlations.index
                if c != target_col
            ][:5]

        except:
            pass

        for feature in strongest_features:

            fig = plt.figure(
                figsize=(7, 5)
            )

            plt.scatter(
                df[feature],
                df[target_col],
                s=12
            )

            plt.xlabel(feature)
            plt.ylabel(target_col)

            plt.title(
                f"{feature} vs {target_col}"
            )

            p = os.path.join(
                CHART,
                f"{idx}_{feature}_target.png"
            )

            fig.savefig(
                p,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            charts.append(p)

            descriptions.append(
                f"Relationship between {feature} and {target_col}."
            )

    # -----------------------------
    # NATO 2% DEFENSE TARGET
    # -----------------------------

    if (
        "meets_2_percent_target"
        in df.columns
    ):

        fig = plt.figure(
            figsize=(7, 5)
        )

        counts = (
            df[
                "meets_2_percent_target"
            ]
            .astype(str)
            .value_counts()
        )

        plt.bar(
            counts.index,
            counts.values
        )

        plt.title(
            "NATO 2% Defense Target Compliance"
        )

        p = os.path.join(
            CHART,
            f"{idx}_target2pct.png"
        )

        fig.savefig(
            p,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(p)

        descriptions.append(
            "Countries meeting NATO's 2% GDP defense target."
        )

    # -----------------------------
    # DEFENSE SPENDING TREND
    # -----------------------------

    if (
        "year" in df.columns
        and
        "defense_budget_billion_usd"
        in df.columns
    ):

        yearly = (
            df.groupby("year")[
                "defense_budget_billion_usd"
            ]
            .mean()
        )

        fig = plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            yearly.index,
            yearly.values
        )

        plt.title(
            "Defense Budget Trend"
        )

        plt.xlabel("Year")
        plt.ylabel(
            "Avg Defense Budget"
        )

        p = os.path.join(
            CHART,
            f"{idx}_budget_trend.png"
        )

        fig.savefig(
            p,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(p)

        descriptions.append(
            "Average NATO defense budget trend over time."
        )

    # -----------------------------
    # NATO EXPANSION TIMELINE
    # -----------------------------

    if "join_year" in df.columns:

        fig = plt.figure(
            figsize=(9, 5)
        )

        join_counts = (
            df[
                "join_year"
            ]
            .dropna()
            .value_counts()
            .sort_index()
        )

        cumulative = (
            join_counts.cumsum()
        )

        plt.plot(
            cumulative.index,
            cumulative.values
        )

        plt.title(
            "NATO Membership Expansion"
        )

        plt.xlabel(
            "Join Year"
        )

        plt.ylabel(
            "Countries Joined"
        )

        p = os.path.join(
            CHART,
            f"{idx}_membership.png"
        )

        fig.savefig(
            p,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(p)

        descriptions.append(
            "Timeline of NATO membership growth."
        )

    # -----------------------------
    # MISSION COST TREND
    # -----------------------------

    if (
        "operation_start_year"
        in df.columns
        and
        "mission_cost_m_usd"
        in df.columns
    ):

        trend = (
            df.groupby(
                "operation_start_year"
            )[
                "mission_cost_m_usd"
            ]
            .mean()
        )

        fig = plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            trend.index,
            trend.values
        )

        plt.title(
            "Mission Cost Trend"
        )

        plt.xlabel("Year")
        plt.ylabel(
            "Mission Cost"
        )

        p = os.path.join(
            CHART,
            f"{idx}_mission_cost.png"
        )

        fig.savefig(
            p,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(p)

        descriptions.append(
            "Average NATO mission cost over time."
        )

    # -----------------------------
    # EQUIPMENT READINESS
    # -----------------------------

    if (
        "combat_ready_pct"
        in df.columns
    ):

        fig = plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            df[
                "combat_ready_pct"
            ],
            bins=30
        )

        plt.title(
            "Combat Readiness Distribution"
        )

        p = os.path.join(
            CHART,
            f"{idx}_combat_ready.png"
        )

        fig.savefig(
            p,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(p)

        descriptions.append(
            "Combat readiness distribution."
        )
    # -----------------------------
    # PREPROCESSING FOR ML
    # -----------------------------

    rmse_rf = None
    rmse_gb = None
    best_k = None
    silhouette = None

    try:

        processed_data, _, _ = (
            preprocess_data(df)
        )

    except:
        processed_data = None

    # -----------------------------
    # KMEANS CLUSTERING
    # -----------------------------

    if processed_data is not None:

        try:

            max_k = min(
                8,
                len(df) - 1
            )

            best_score = -1

            for k in range(2, max_k):

                km = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=42
                )

                labels = km.fit_predict(
                    processed_data
                )

                score = silhouette_score(
                    processed_data,
                    labels
                )

                if score > best_score:

                    best_score = score
                    best_k = k
                    silhouette = score

            final_kmeans = KMeans(
                n_clusters=best_k,
                n_init=10,
                random_state=42
            )

            df["cluster"] = (
                final_kmeans.fit_predict(
                    processed_data
                )
            )

            fig = plt.figure(
                figsize=(7, 5)
            )

            cluster_counts = (
                pd.Series(
                    df["cluster"]
                )
                .value_counts()
                .sort_index()
            )

            plt.bar(
                cluster_counts.index.astype(str),
                cluster_counts.values
            )

            plt.title(
                f"KMeans Clusters (k={best_k})"
            )

            cluster_path = os.path.join(
                CHART,
                f"{idx}_cluster.png"
            )

            fig.savefig(
                cluster_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            charts.append(
                cluster_path
            )

            descriptions.append(
                f"Validated KMeans clustering using silhouette optimization (k={best_k})."
            )

        except:
            pass

    # -----------------------------
    # ANOMALY DETECTION
    # -----------------------------

    if processed_data is not None:

        try:

            iso = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            df["anomaly"] = (
                iso.fit_predict(
                    processed_data
                )
            )

            fig = plt.figure(
                figsize=(7, 5)
            )

            anomaly_counts = (
                pd.Series(
                    df["anomaly"]
                )
                .value_counts()
            )

            plt.bar(
                anomaly_counts.index.astype(str),
                anomaly_counts.values
            )

            plt.title(
                "Anomaly Detection"
            )

            anomaly_path = os.path.join(
                CHART,
                f"{idx}_anomaly.png"
            )

            fig.savefig(
                anomaly_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            charts.append(
                anomaly_path
            )

            descriptions.append(
                "Isolation Forest anomaly detection identifying unusual observations."
            )

        except:
            pass

    # -----------------------------
    # MACHINE LEARNING
    # -----------------------------

    feature_importance_path = None

    try:

        if (
            target_col
            and
            target_col in df.columns
        ):

            y = df[target_col]

            X = df.drop(
                columns=[
                    target_col
                ],
                errors="ignore"
            )

            numeric_features = (
                X.select_dtypes(
                    include=np.number
                )
                .columns
                .tolist()
            )

            categorical_features = (
                X.select_dtypes(
                    include="object"
                )
                .columns
                .tolist()
            )

            numeric_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = (
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            numeric_transformer,
                            numeric_features
                        ),
                        (
                            "cat",
                            categorical_transformer,
                            categorical_features
                        )
                    ]
                )
            )

            X_processed = (
                preprocessor
                .fit_transform(X)
            )

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X_processed,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

            rf = (
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )

            rf.fit(
                X_train,
                y_train
            )

            pred_rf = rf.predict(
                X_test
            )

            rmse_rf = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_rf
                )
            )

            gb = (
                GradientBoostingRegressor(
                    random_state=42
                )
            )

            gb.fit(
                X_train,
                y_train
            )

            pred_gb = gb.predict(
                X_test
            )

            rmse_gb = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_gb
                )
            )

            # -------------------------
            # FEATURE IMPORTANCE
            # -------------------------

            try:

                feature_names = (
                    numeric_features
                )

                importances = (
                    rf.feature_importances_
                    [:len(feature_names)]
                )

                imp_df = pd.DataFrame({
                    "feature":
                    feature_names,
                    "importance":
                    importances
                })

                imp_df = (
                    imp_df
                    .sort_values(
                        "importance",
                        ascending=False
                    )
                    .head(10)
                )

                fig = plt.figure(
                    figsize=(8, 5)
                )

                plt.barh(
                    imp_df[
                        "feature"
                    ],
                    imp_df[
                        "importance"
                    ]
                )

                plt.title(
                    "Top Feature Importance"
                )

                feature_importance_path = (
                    os.path.join(
                        CHART,
                        f"{idx}_importance.png"
                    )
                )

                fig.savefig(
                    feature_importance_path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                charts.append(
                    feature_importance_path
                )

                descriptions.append(
                    "Random Forest feature importance."
                )

            except:
                pass

    except:
        pass

    # -----------------------------
    # SUMMARY
    # -----------------------------

    summary = f"""
    <b>Rows:</b> {len(df)}<br/>
    <b>Columns:</b> {len(df.columns)}<br/>
    <b>Target Variable:</b> {target_col}<br/>
    <b>Numerical Features:</b> {len(numeric_cols)}<br/>
    <b>Categorical Features:</b> {len(categorical_cols)}<br/>
    <b>Generated Charts:</b> {len(charts)}<br/>
    <b>Optimal Clusters:</b> {best_k if best_k else 'N/A'}<br/>
    <b>Silhouette Score:</b> {round(silhouette,3) if silhouette else 'N/A'}<br/>
    <b>Random Forest RMSE:</b> {round(rmse_rf,3) if rmse_rf else 'N/A'}<br/>
    <b>Gradient Boosting RMSE:</b> {round(rmse_gb,3) if rmse_gb else 'N/A'}<br/>
    """

    elements.append(
        Paragraph(
            summary,
            body_style
        )
    )

    elements.append(
        Spacer(
            1,
            0.2 * inch
        )
    )

    # -----------------------------
    # ADD CHARTS TO PDF
    # -----------------------------

    for i in range(
        len(charts)
    ):

        try:

            elements.append(
                Image(
                    charts[i],
                    width=6.2 * inch,
                    height=4.0 * inch
                )
            )

            elements.append(
                Paragraph(
                    descriptions[i],
                    body_style
                )
            )

            elements.append(
                Spacer(
                    1,
                    0.15 * inch
                )
            )

        except:
            pass

    elements.append(
        PageBreak()
    )

    all_summary.append({
        "dataset":
        dataset_name,
        "rows":
        len(df),
        "columns":
        len(df.columns),
        "target":
        target_col,
        "charts":
        len(charts)
    })


# ---------------------------------
# FINAL SUMMARY PAGE
# ---------------------------------

final_summary = (
    "<b>Processed Datasets:</b><br/><br/>"
)

for s in all_summary:

    final_summary += f"""
    • {s['dataset']}<br/>
    Rows: {s['rows']} |
    Columns: {s['columns']} |
    Target: {s['target']} |
    Charts: {s['charts']}<br/><br/>
    """

elements.append(
    Paragraph(
        "Final Dataset Summary",
        heading_style
    )
)

elements.append(
    Paragraph(
        final_summary,
        body_style
    )
)

doc.build(elements)

print(
    "Datasets Processed:",
    len(all_summary)
)

print(
    "PDF Report Generated:",
    report_path
)

print(
    "Execution Time:",
    round(
        time.time() - start,
        2
    ),
    "seconds"
)
