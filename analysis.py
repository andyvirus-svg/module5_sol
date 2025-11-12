import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('pymc').setLevel(logging.ERROR)

df1 = pd.read_csv('df_1.csv', index_col=0)
df2 = pd.read_csv('df_2.csv', index_col=0)
df3 = pd.read_csv('df_3.csv', index_col=0)

df1['Status'] = 'Active'
df2['Status'] = 'Defunct'
df3['Status'] = 'Discontinued'

df_merged = pd.concat([df1, df2, df3], ignore_index=True)
df_merged['Region'] = df_merged['Region'].fillna('Unknown')
df_merged = df_merged[df_merged['Region'] != 'Unknown']

print("BAYESIAN HIERARCHICAL MODEL - REGIONAL SURVIVAL PROBABILITY")

region_data = df_merged.groupby('Region').agg({
    'Status': lambda x: (x == 'Active').sum(),
    'Brand': 'count'
}).reset_index()
region_data.columns = ['Region', 'Active', 'Total']

try:
    import pymc as pm
    import arviz as az

    region_idx = {region: i for i, region in enumerate(region_data['Region'])}
    region_indices = np.array([region_idx[r] for r in region_data['Region']])
    n_regions = len(region_idx)

    with pm.Model() as hierarchical_model:
        mu_logit_p = pm.Normal('mu_logit_p', mu=2, sigma=2)
        sigma_logit_p = pm.HalfNormal('sigma_logit_p', sigma=1)

        logit_p_offset = pm.Normal('logit_p_offset', mu=0, sigma=1, shape=n_regions)
        logit_p_region = pm.Deterministic('logit_p_region', mu_logit_p + logit_p_offset * sigma_logit_p)

        p = pm.Deterministic('p', pm.math.invlogit(logit_p_region))

        y = pm.Binomial('y', n=region_data['Total'].values, p=p[region_indices],
                        observed=region_data['Active'].values)

        trace = pm.sample(3000, tune=3000, target_accept=0.99,
                         return_inferencedata=True, random_seed=42,
                         progressbar=False, cores=1)

    posterior_means = trace.posterior['p'].mean(dim=['chain', 'draw']).values

    asia_idx = region_idx['Asia']
    europe_idx = region_idx['Europe']

    asia_prob = posterior_means[asia_idx]
    europe_prob = posterior_means[europe_idx]

except ImportError:
    print("Installing PyMC...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pymc', '--break-system-packages'])
    subprocess.check_call(['pip', 'install', 'arviz', '--break-system-packages'])

    import pymc as pm
    import arviz as az

    region_idx = {region: i for i, region in enumerate(region_data['Region'])}
    region_indices = np.array([region_idx[r] for r in region_data['Region']])
    n_regions = len(region_idx)

    with pm.Model() as hierarchical_model:
        mu_logit_p = pm.Normal('mu_logit_p', mu=2, sigma=2)
        sigma_logit_p = pm.HalfNormal('sigma_logit_p', sigma=1)

        logit_p_offset = pm.Normal('logit_p_offset', mu=0, sigma=1, shape=n_regions)
        logit_p_region = pm.Deterministic('logit_p_region', mu_logit_p + logit_p_offset * sigma_logit_p)

        p = pm.Deterministic('p', pm.math.invlogit(logit_p_region))

        y = pm.Binomial('y', n=region_data['Total'].values, p=p[region_indices],
                        observed=region_data['Active'].values)

        trace = pm.sample(3000, tune=3000, target_accept=0.99,
                         return_inferencedata=True, random_seed=42,
                         progressbar=False, cores=1)

    posterior_means = trace.posterior['p'].mean(dim=['chain', 'draw']).values

    asia_idx = region_idx['Asia']
    europe_idx = region_idx['Europe']

    asia_prob = posterior_means[asia_idx]
    europe_prob = posterior_means[europe_idx]

print(f"Posterior mean survival probability for Asia: {asia_prob:.4f}")
print(f"Posterior mean survival probability for Europe: {europe_prob:.4f}")

print("\nMEDIATION ANALYSIS")

country_stats = df_merged.groupby('Country').agg({
    'Brand': 'count',
    'Status': lambda x: ((x == 'Defunct') | (x == 'Discontinued')).sum(),
    'Region': lambda x: x.nunique()
}).reset_index()
country_stats.columns = ['Country', 'Brand_Count', 'Exit_Count', 'Region_Diversity']
country_stats['Exit_Rate'] = country_stats['Exit_Count'] / country_stats['Brand_Count']

country_stats = country_stats[country_stats['Brand_Count'] > 1].copy()

X = country_stats['Region_Diversity'].values.reshape(-1, 1)
M = country_stats['Brand_Count'].values.reshape(-1, 1)
Y = country_stats['Exit_Rate'].values.reshape(-1, 1)

model_a = LinearRegression()
model_a.fit(X, M)
a_coef = model_a.coef_[0][0]

X_M = np.hstack([X, M])
model_b = LinearRegression()
model_b.fit(X_M, Y)
b_coef = model_b.coef_[0][1]

model_c = LinearRegression()
model_c.fit(X, Y)
c_coef = model_c.coef_[0][0]

indirect_effect = a_coef * b_coef
total_effect = c_coef
proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0

print(f"Indirect effect coefficient: {indirect_effect:.4f}")
print(f"Proportion of total effect mediated: {proportion_mediated:.4f}")

print("\nGAUSSIAN MIXTURE MODEL - MARKET HEALTH STATES")

country_features = country_stats[['Brand_Count', 'Exit_Rate']].values

gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
gmm.fit(country_features)

hidden_states = gmm.predict(country_features)
state_probs = gmm.predict_proba(country_features)

state_means = []
for i in range(3):
    state_indices = hidden_states == i
    if state_indices.sum() > 0:
        avg_brands = country_features[state_indices, 0].mean()
        avg_exit = country_features[state_indices, 1].mean()
        health_score = avg_brands * (1 - avg_exit)
        state_means.append((i, health_score))

state_means.sort(key=lambda x: x[1], reverse=True)
healthiest_state = state_means[0][0]

steady_state_prob = gmm.weights_[healthiest_state]

healthiest_country_idx = state_probs[:, healthiest_state].argmax()
healthiest_country = country_stats.iloc[healthiest_country_idx]['Country']

print(f"Steady-state probability of healthiest state: {steady_state_prob:.4f}")
print(f"Country with highest probability in healthiest state: {healthiest_country}")

print("\nGAUSSIAN PROCESS REGRESSION")

X_gp = country_stats['Brand_Count'].values.reshape(-1, 1)
y_gp = country_stats['Exit_Rate'].values

X_train, X_test, y_train, y_test = train_test_split(X_gp, y_gp, test_size=0.3, random_state=42)

kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
gp.fit(X_train, y_train)

y_pred, y_std = gp.predict(X_test, return_std=True)

nlpd_values = []
for i in range(len(y_test)):
    pred_mean = y_pred[i]
    pred_std = y_std[i]
    true_val = y_test[i]

    log_likelihood = -0.5 * np.log(2 * np.pi * pred_std**2) - 0.5 * ((true_val - pred_mean)**2 / pred_std**2)
    nlpd_values.append(-log_likelihood)

nlpd = np.mean(nlpd_values)

print(f"Test set negative log predictive density: {nlpd:.4f}")

print("\nVISUALIZATION 1: ALLUVIAL DIAGRAM")

top_countries = df_merged.groupby('Country')['Brand'].count().nlargest(5).index.tolist()
df_alluvial = df_merged[df_merged['Country'].isin(top_countries)].copy()

# Ensure plotly is installed
try:
    import plotly.graph_objects as go
except ImportError:
    print("Installing Plotly...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly', '--break-system-packages'])
    import plotly.graph_objects as go

region_country = df_alluvial.groupby(['Region', 'Country']).size().reset_index(name='count')
country_status = df_alluvial.groupby(['Country', 'Status']).size().reset_index(name='count')

regions = region_country['Region'].unique().tolist()
countries = df_alluvial['Country'].unique().tolist()
statuses = df_alluvial['Status'].unique().tolist()

all_nodes = regions + countries + statuses
node_dict = {node: i for i, node in enumerate(all_nodes)}

sources = []
targets = []
values = []

for _, row in region_country.iterrows():
    sources.append(node_dict[row['Region']])
    targets.append(node_dict[row['Country']])
    values.append(row['count'])

for _, row in country_status.iterrows():
    sources.append(node_dict[row['Country']])
    targets.append(node_dict[row['Status']])
    values.append(row['count'])

status_colors = {'Active': 'green', 'Defunct': 'red', 'Discontinued': 'orange'}
node_colors = []
for node in all_nodes:
    if node in statuses:
        node_colors.append(status_colors[node])
    else:
        node_colors.append('lightblue')

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
)])

fig.update_layout(
    title_text="Mobile Phone Manufacturer Flow: Region → Country → Status",
    font_size=12,
    height=600
)

# Save as HTML and try PNG if kaleido is available
fig.write_html("alluvial_diagram.html")
print("✓ Alluvial diagram saved as 'alluvial_diagram.html'")

try:
    fig.write_image("alluvial_diagram.png", width=1200, height=600, scale=2)
    print("✓ Alluvial diagram also saved as 'alluvial_diagram.png'")
except Exception as e:
    print("  (PNG export skipped - kaleido not available)")

fig.show()

print("\nVISUALIZATION 2: TERNARY PLOT")

region_composition = []
for region in df_merged['Region'].unique():
    region_df = df_merged[df_merged['Region'] == region]
    total = len(region_df)

    active_pct = (region_df['Status'] == 'Active').sum() / total
    defunct_pct = (region_df['Status'] == 'Defunct').sum() / total
    discontinued_pct = (region_df['Status'] == 'Discontinued').sum() / total

    region_composition.append({
        'Region': region,
        'Active': active_pct,
        'Defunct': defunct_pct,
        'Discontinued': discontinued_pct,
        'Total': total
    })

comp_df = pd.DataFrame(region_composition)

fig = go.Figure(go.Scatterternary(
    a=comp_df['Active'],
    b=comp_df['Defunct'],
    c=comp_df['Discontinued'],
    text=comp_df['Region'],
    mode='markers+text',
    marker=dict(
        size=10 + comp_df['Total'] / 2,
        color=list(range(len(comp_df))),
        colorscale='Viridis',
        line=dict(width=2, color='black'),
        showscale=False
    ),
    textposition="top center",
    textfont=dict(size=11, color='black')
))

fig.update_layout(
    title='Ternary Plot: Regional Manufacturer Status Composition',
    ternary=dict(
        sum=1,
        aaxis=dict(title='Active (%)', min=0, linewidth=2, ticks='outside'),
        baxis=dict(title='Defunct (%)', min=0, linewidth=2, ticks='outside'),
        caxis=dict(title='Discontinued (%)', min=0, linewidth=2, ticks='outside')
    ),
    height=700, width=900, showlegend=False
)

# Save as HTML and try PNG if kaleido is available
fig.write_html("ternary_plot.html")
print("✓ Ternary plot saved as 'ternary_plot.html'")

try:
    fig.write_image("ternary_plot.png", width=1200, height=900, scale=2)
    print("✓ Ternary plot also saved as 'ternary_plot.png'")
except Exception as e:
    print("  (PNG export skipped - kaleido not available)")

fig.show()

print("\nSTRATEGIC DECISION")

survival_diff = asia_prob - europe_prob
mediation_threshold = abs(proportion_mediated)

condition_1 = survival_diff > 0.10
condition_2 = mediation_threshold > 0.30

strategic_decision = "YES" if (condition_1 and condition_2) else "NO"

print(f"Strategic Question Answer: {strategic_decision}")
