import os
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
import warnings

warnings.filterwarnings("ignore")

start = time.time()

BASE_DIR = os.getcwd()

INPUT_BASE = BASE_DIR

OUT = os.path.join(BASE_DIR, "Output")
CHART = os.path.join(OUT, "charts")

os.makedirs(CHART, exist_ok=True)

csv_files = []

for root, dirs, files in os.walk(INPUT_BASE):
    for f in files:
        path = os.path.join(root, f)

        if f.endswith(".csv"):
            csv_files.append(path)

        elif f.endswith(".zip"):
            extract_dir = os.path.join(BASE_DIR, "unzipped")
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(path, "r") as z:
                z.extractall(extract_dir)

            for zr, zd, zf in os.walk(extract_dir):
                for zz in zf:
                    if zz.endswith(".csv"):
                        csv_files.append(os.path.join(zr, zz))

if len(csv_files) == 0:
    raise ValueError(f"No CSV files found in: {INPUT_BASE}")

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "title",
    fontSize=26,
    alignment=TA_CENTER,
    textColor=HexColor("#22d3ee"),
    spaceAfter=20
)

subtitle_style = ParagraphStyle(
    "subtitle",
    fontSize=16,
    alignment=TA_CENTER,
    textColor=HexColor("#a78bfa"),
    spaceAfter=30
)

body_style = ParagraphStyle(
    "body",
    fontSize=11,
    leading=16,
    spaceAfter=12
)

heading_style = ParagraphStyle(
    "heading",
    fontSize=18,
    textColor=HexColor("#2563eb"),
    spaceAfter=16
)

report_path = os.path.join(OUT, "NATO_Complete_Analysis_Report.pdf")

doc = SimpleDocTemplate(
    report_path,
    pagesize=A4,
    leftMargin=40,
    rightMargin=40,
    topMargin=40,
    bottomMargin=40
)

elements = []

elements.append(Paragraph("NATO Alliance Complete Dataset 2024", title_style))
elements.append(Paragraph("Comprehensive Data Science and Machine Learning Analysis", subtitle_style))

intro = """
This report presents automated exploratory data analysis, machine learning insights,
clustering analysis, anomaly detection, and statistical visualizations generated from
the NATO Alliance Complete Dataset 2024.
"""

elements.append(Paragraph(intro, body_style))
elements.append(PageBreak())

all_summary = []

for idx, file_path in enumerate(csv_files):

    dataset_name = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path)
    except:
        continue

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    categorical = df.select_dtypes(include="object").columns

    for c in categorical:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    numerical = df.select_dtypes(include=np.number).columns

    if len(numerical) == 0:
        continue

    df[numerical] = df[numerical].replace([np.inf, -np.inf], np.nan)
    df[numerical] = df[numerical].fillna(df[numerical].median())
    df = df.dropna()

    core = None

    priority_targets = [
        "gdp_billion_usd",
        "defense_budget_billion_usd",
        "mission_cost_m_usd",
        "troops_deployed",
        "units_count",
        "total_value_m_usd"
    ]

    for t in priority_targets:
        if t in df.columns:
            core = t
            break

    if core is None:
        core = numerical[0]

    charts = []
    descriptions = []

    corr_fig = plt.figure(figsize=(12, 10))
    corr = df[numerical].corr()

    plt.imshow(corr, aspect="auto")
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title(f"Correlation Matrix - {dataset_name}")

    corr_path = os.path.join(CHART, f"corr_{idx}.png")

    corr_fig.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()

    charts.append(corr_path)

    descriptions.append(
        "Correlation analysis between numerical variables showing positive and negative feature relationships."
    )

    limited_num = list(numerical[:8])

    for col in limited_num:

        fig = plt.figure(figsize=(8, 5))

        plt.hist(df[col], bins=30)

        plt.title(f"{col} Distribution")

        p = os.path.join(CHART, f"{idx}_{col}_hist.png")

        fig.savefig(p, dpi=300, bbox_inches="tight")

        plt.close()

        charts.append(p)

        descriptions.append(
            f"Distribution analysis for {col} showing value spread and density patterns."
        )

    pair_counter = 0

    for i in range(len(limited_num)):
        for j in range(i + 1, len(limited_num)):

            fig = plt.figure(figsize=(7, 5))

            plt.scatter(df[limited_num[i]], df[limited_num[j]], s=10)

            plt.xlabel(limited_num[i])
            plt.ylabel(limited_num[j])

            plt.title(f"{limited_num[i]} vs {limited_num[j]}")

            p = os.path.join(CHART, f"{idx}_pair_{i}_{j}.png")

            fig.savefig(p, dpi=300, bbox_inches="tight")

            plt.close()

            charts.append(p)

            descriptions.append(
                f"Scatter relationship between {limited_num[i]} and {limited_num[j]}."
            )

            pair_counter += 1

            if pair_counter >= 10:
                break

        if pair_counter >= 10:
            break

    feature_data = df[numerical]

    try:
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(feature_data)

        fig = plt.figure(figsize=(7, 5))

        unique, counts = np.unique(df["cluster"], return_counts=True)

        plt.bar(unique.astype(str), counts)

        plt.title("Cluster Distribution")

        p = os.path.join(CHART, f"{idx}_clusters.png")

        fig.savefig(p, dpi=300, bbox_inches="tight")

        plt.close()

        charts.append(p)

        descriptions.append(
            "KMeans clustering groups similar records into data-driven clusters."
        )

    except:
        pass

    try:
        iso = IsolationForest(contamination=0.05, random_state=42)

        df["anomaly"] = iso.fit_predict(feature_data)

        fig = plt.figure(figsize=(7, 5))

        unique, counts = np.unique(df["anomaly"], return_counts=True)

        plt.bar(unique.astype(str), counts)

        plt.title("Anomaly Detection")

        p = os.path.join(CHART, f"{idx}_anomaly.png")

        fig.savefig(p, dpi=300, bbox_inches="tight")

        plt.close()

        charts.append(p)

        descriptions.append(
            "Isolation Forest identifies unusual or abnormal observations within the dataset."
        )

    except:
        pass

    rmse_rf = None
    rmse_gb = None

    try:
        X = feature_data.drop(columns=[core], errors="ignore")
        y = df[core]

        if len(X.columns) > 0:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            rf = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )

            rf.fit(X_train, y_train)

            pred_rf = rf.predict(X_test)

            rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))

            gb = GradientBoostingRegressor(random_state=42)

            gb.fit(X_train, y_train)

            pred_gb = gb.predict(X_test)

            rmse_gb = np.sqrt(mean_squared_error(y_test, pred_gb))

    except:
        pass

    section_title = f"Dataset Analysis: {dataset_name}"

    elements.append(Paragraph(section_title, heading_style))

    summary = f"""
    <b>Rows:</b> {len(df)}<br/>
    <b>Columns:</b> {len(df.columns)}<br/>
    <b>Target Variable:</b> {core}<br/>
    <b>Numerical Features:</b> {len(numerical)}<br/>
    <b>Generated Charts:</b> {len(charts)}<br/>
    <b>Random Forest RMSE:</b> {round(rmse_rf, 3) if rmse_rf is not None else "N/A"}<br/>
    <b>Gradient Boosting RMSE:</b> {round(rmse_gb, 3) if rmse_gb is not None else "N/A"}<br/>
    """

    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 0.2 * inch))

    for i in range(len(charts)):

        try:
            elements.append(
                Image(
                    charts[i],
                    width=6.2 * inch,
                    height=4.2 * inch
                )
            )

            elements.append(
                Paragraph(descriptions[i], body_style)
            )

            elements.append(Spacer(1, 0.15 * inch))

        except:
            pass

    elements.append(PageBreak())

    all_summary.append({
        "dataset": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "target": core,
        "charts": len(charts)
    })

final_summary = "<b>Processed Datasets:</b><br/><br/>"

for s in all_summary:

    final_summary += f"""
    • {s['dataset']}<br/>
    Rows: {s['rows']} |
    Columns: {s['columns']} |
    Target: {s['target']} |
    Charts: {s['charts']}<br/><br/>
    """

elements.append(Paragraph("Final Dataset Summary", heading_style))
elements.append(Paragraph(final_summary, body_style))

doc.build(elements)

print("Datasets Processed:", len(all_summary))
print("PDF Report Generated:", report_path)
print("Execution Time:", round(time.time() - start, 2), "seconds")