# 🌍 NATO Alliance Complete Dataset 2024

![NATO](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Flag_of_NATO.svg/320px-Flag_of_NATO.svg.png)

## 📌 About This Dataset

A comprehensive, cleaned dataset covering **all 32 NATO member countries** with three interconnected files spanning **10,700+ rows** of data on country statistics, military equipment, and NATO operations/missions from **1949 to 2024**.

---

## 📁 Files Overview

### 1. `NATO_1_Country_Stats.csv` — 1,444 rows | 27 columns
Yearly economic and military statistics for each NATO member country.

| Column | Description |
|--------|-------------|
| Record_ID | Unique row identifier |
| Country | Full country name |
| ISO_Code | 3-letter ISO country code |
| Join_Year | Year the country joined NATO |
| Years_In_NATO | Years as a member at that year |
| Founding_Member | Yes/No — original 1949 member |
| Nuclear_Sharing | Part of NATO nuclear sharing agreement |
| Region | Geographic region |
| Capital | Capital city |
| Area_km2 | Country area in square kilometers |
| Government_Type | Type of government |
| Alliance_Role | Role within the alliance |
| Year | Data year (1949–2024) |
| Population_M | Population in millions |
| GDP_Billion_USD | Gross Domestic Product (USD Billion) |
| GDP_Per_Capita_USD | GDP per person in USD |
| Inflation_Rate_Pct | Annual inflation rate (%) |
| Unemployment_Rate_Pct | Unemployment rate (%) |
| Defense_Budget_Billion_USD | Annual defense spending (USD Billion) |
| Defense_GDP_Percent | Defense spending as % of GDP |
| Meets_2_Percent_Target | Meets NATO's 2% GDP target (Yes/No) |
| Active_Military_Personnel | Number of active duty soldiers |
| Reserve_Personnel | Number of reserve soldiers |
| Total_Military_Personnel | Total soldiers (active + reserve) |
| NATO_Contribution_Rank | Relative contribution rank (1=highest) |
| Interoperability_Score | NATO interoperability score (0–100) |
| Training_Exercises_Per_Year | Annual training exercises count |

---

### 2. `NATO_2_Equipment_Inventory.csv` — 4,056 rows | 25 columns
Military equipment inventory across all NATO countries.

| Column | Description |
|--------|-------------|
| Record_ID | Unique row identifier |
| Country | Country name |
| ISO_Code | ISO country code |
| Equipment_Type | Name of equipment (e.g. Main Battle Tank) |
| Equipment_Category | Category (Armored, Artillery, Naval, etc.) |
| Domain | Land / Air / Sea / Digital |
| Notable_Models | Real-world model examples |
| Units_Count | Number of units in inventory |
| Operational_Status | Operational / Maintenance / Reserve / etc. |
| Condition | Excellent / Good / Fair / etc. |
| Year_Acquired | Year the equipment was procured |
| Equipment_Age_Years | Age of equipment in years |
| Unit_Cost_M_USD | Cost per unit (USD Million) |
| Total_Value_M_USD | Total inventory value (USD Million) |
| Country_of_Origin | Where equipment was manufactured |
| NATO_Standardized | Standardized across NATO? (Yes/No/Partial) |
| Interoperable | Interoperable with allied systems? |
| Last_Maintenance_Year | Year of last maintenance |
| Next_Upgrade_Due | Scheduled upgrade year |
| Combat_Ready_Pct | % of units combat-ready |

---

### 3. `NATO_3_Operations_Missions.csv` — 5,200 rows | 30 columns
Full log of NATO operations, missions, and exercises.

| Column | Description |
|--------|-------------|
| Record_ID | Unique row identifier |
| Mission_Name | Operation name |
| Mission_Type | Type (Combat / Peacekeeping / Training / etc.) |
| Lead_Country | Country leading the operation |
| Operation_Location | Geographic location of operation |
| Operation_Region | Broader region |
| Threat_Level | High / Medium / Low |
| Command_HQ | NATO command headquarters |
| Operation_Start_Year | Year operation started |
| Operation_End_Year | Year operation ended |
| Duration_Years | Duration in years |
| Mission_Phase | Planning / Deployment / Execution / etc. |
| Troops_Deployed | Number of troops deployed |
| Air_Assets_Deployed | Number of aircraft deployed |
| Naval_Assets_Deployed | Number of naval vessels deployed |
| Casualties | Total casualties reported |
| Casualties_Rate_Pct | Casualties as % of troops deployed |
| Mission_Cost_M_USD | Total mission cost (USD Million) |
| Cost_Per_Soldier_USD | Cost per soldier deployed |
| Contributing_Countries_Count | Number of contributing nations |
| NATO_Led | Was it NATO-led? (Yes/No/Joint) |
| UN_Mandate | Did it have a UN mandate? |
| Mission_Outcome | Successful / Completed / Ongoing / etc. |
| Mission_Status | Active / Completed / Suspended / etc. |
| Classification | Unclassified / Restricted / etc. |
| Media_Coverage | High / Medium / Low / Minimal |
| Public_Support_Pct | Public approval rating (%) |
| After_Action_Report | Published / Classified / Pending |

---

## 💡 Possible Use Cases

- 📊 **Defense spending trend analysis** — Which countries meet the 2% GDP target?
- 🗺️ **Geopolitical research** — NATO expansion over time
- 🔫 **Military capability comparison** — Equipment inventory by country
- 🎯 **Mission analysis** — Success rates, costs, and outcomes
- 📈 **Time-series analysis** — GDP and military trends (1949–2024)
- 🤖 **ML classification** — Predict mission outcomes
- 📉 **Regression modeling** — Defense budget vs. GDP growth

---

## 🔑 Key Facts

- **32 NATO member countries** represented
- **10,700+ total rows** across 3 files
- **82 total columns** (combined)
- **Time range: 1949–2024** (75 years)
- **Zero empty cells** — fully cleaned dataset

---

## ⚠️ Disclaimer

This dataset contains **simulated/synthetic data** generated for educational, research, and data analysis practice purposes. Values are statistically realistic but not official NATO figures. For official data, refer to [NATO official statistics](https://www.nato.int/nato_static_fl2014/assets/pdf/2024/3/pdf/240314-def-exp-2023-en.pdf).

---

## 📜 License

This dataset is released under the **CC0 1.0 Universal (Public Domain)** license.
You are free to copy, modify, and distribute this data, even for commercial purposes, without asking permission.

---

## 🙏 Acknowledgements

- [NATO Official Website](https://www.nato.int)
- [SIPRI Military Expenditure Database](https://www.sipri.org)
- [World Bank Open Data](https://data.worldbank.org)
