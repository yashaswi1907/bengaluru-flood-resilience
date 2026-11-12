# Prompts for the remaining modules

Copy a prompt into Claude together with this project folder (or `README.md` + `src/predict.py` + `outputs/cell_risk_scores.csv`). Every module builds on `FloodPredictor` from Module 1.

---

## Module 2 – Infrastructure Intelligence
> I have a Bengaluru flood-risk model. `outputs/cell_risk_scores.csv` gives flood_probability per ~1 km cell (cell_id, lat, lon, zone) for 50/100/150/200 mm rainfall. Use osmnx to download hospitals, schools, fire stations, police stations, metro stations and bus depots for Bengaluru. Map each asset to its nearest cell, and output a table of infrastructure at risk per scenario (asset, type, cell risk, severity). Also produce a map overlay and a "vulnerable locations" list ranked by risk × importance.

## Module 3 – Impact & Network Resilience
> Using osmnx, build Bengaluru's drivable road graph (nodes = intersections, edges = roads). Using `FloodPredictor.scenario(rain)` from `src/predict.py`, mark an edge as closed when its cell's flood_probability ≥ the model threshold. Compute: % travel-time increase between 200 random OD pairs, disconnected components, population that loses access to a hospital within 15 minutes, and edge betweenness to find critical roads. Compare 100 mm vs 150 mm vs 200 mm.

## Module 4 – Emergency Response & Resource Allocation
> Given cell-level flood risk, population density, road-network accessibility loss (Module 3) and hospital distance, build a priority score for each cell (state the weights). Then allocate K rescue teams / pumps to depots using a p-median or max-coverage model in PuLP so as to maximise population covered within 10 minutes on the flooded network. Output the priority zones and the deployment plan.

## Module 5 – Resilience & Intervention Optimisation
> `FloodPredictor.scenario()` accepts drain_capacity_boost_pct, desilting and green_cover_increase. Define interventions per zone (drain upgrade, desilting, culvert upgrade, lake restoration, green infrastructure, pumping) with costs in ₹ crore. Use PuLP to choose the set of interventions that maximises the reduction in expected people at risk under a ₹25–100 crore budget. Show a cost vs. benefit chart and the Pareto front for budget vs. risk reduction.

## Module 6 – What-if Simulator
> Build a Streamlit page with sliders: rainfall 50–200 mm, drainage capacity upgrade %, budget ₹25–100 crore, and intervention type. On change, re-run `FloodPredictor.scenario()` and the Module 3 network analysis, then show flood risk, road disruptions, population affected, critical infrastructure at risk and an overall resilience score.

## Module 7 – Decision-Support Dashboard
> Create a Streamlit dashboard with two modes. Planner mode: risk maps (pydeck/folium), vulnerable roads, impact analysis, intervention recommendations, budget optimisation, report export. Public mode: my risk zone (search by locality), roads to avoid, nearest safe hospital route, safe routes. Use the outputs of Modules 1–6.

## Report / presentation
> Write the methodology and results section of our project report using README.md, DATA_SOURCES.md and outputs/metrics.json. Explain the storm-wise GroupKFold and time-based test split, why PR-AUC/F1 matter more than accuracy here, and the validation against BBMP hotspots. State plainly which data layers are real and which are simulated.
