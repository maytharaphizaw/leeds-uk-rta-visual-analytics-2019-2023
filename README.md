# leeds-uk-rta-visual-analytics-2019-2023
Visual analytics of Road Traffic Accidents (Leeds, 2019–2023): Python preprocessing &amp; statistical workflow, Tableau dashboards, event/weather/temporal/demographic/infrastructure insights.



# Visualisation of Road Traffic Accidents in Leeds (2019–2023)

**MSc Data Science Dissertation – Leeds Beckett University**  
Author: _May Tharaphi Zaw_

> Interactive, policy-ready visual analytics of road traffic accidents in **Leeds (2019–2023)**.  
> Python for preprocessing/EDA, **Tableau dashboards** for events, weather, temporal, demographics, and infrastructure insights.

---

## Overview

Road Traffic Accidents (RTAs) are a major public health issue. This project integrates **STATS19** collision data with **weather** (rainfall & sunshine) and **Leeds United matchdays** to study:
- Event-related risk (±3 hours around kick-off, **7 km** radius of Elland Road),
- Weather effects (light rain & glare),
- Temporal patterns (rush hours, Fridays, COVID dips),
- Demographics (male drivers, 11–15 pedestrians, cyclists),
- Infrastructure (single carriageways, uncontrolled junctions).

Outputs are delivered via **interactive Tableau dashboards** to support evidence-based road safety decisions.

---

## Tech Stack

- **Python** (pandas, numpy, statsmodels, geopy, folium, tqdm, matplotlib)
- **Tableau** (interactive dashboards, calculated fields)
- **Git LFS** (for `.twbx`, `.pptx`, large CSVs)
- **Conda** environment

---


---

## 🗃️ Data Sources

- **STATS19** (collisions, casualties, vehicles), UK last 5 years.
- **Weather**: Open-Meteo hourly rainfall & sunshine duration.
- **Events**: Leeds United home fixtures (2019-2023).

> _Note:_ Large/raw datasets are **not** committed. See `data/README.md` for how to obtain and place files locally.

---

## 🧪 Reproducibility

### 1) Create environment
```bash
# Option A: pip
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate leeds-rta

