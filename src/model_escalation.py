from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'data' / 'processed' / 'country_month_panel_sample.parquet'
df = pd.read_parquet(path)

# Production version: build a complete country-month calendar first, then create
# a genuinely forward-looking 6-month outcome. This sample demonstrates the design.
df['future_fatalities_6m'] = df.groupby('country')['fatalities'].transform(lambda s: s.shift(-1).rolling(6,min_periods=1).mean())
df['escalation_6m'] = (df['future_fatalities_6m'] > (df['fatalities_6m'] + 1) * 1.5 - 1).astype(int)
features = ['events_log1p','fatalities_log1p','civilian_fatalities_log1p','events_6m','fatalities_6m','civilian_fatalities_6m','fatality_momentum']
X = df[features].replace([np.inf,-np.inf],np.nan)
y = df['escalation_6m'].astype(int)
g = df['country']

models = {
 'logistic': make_pipeline(SimpleImputer(strategy='median'), LogisticRegression(max_iter=2000,class_weight='balanced')),
 'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1))
}
results={}
if g.nunique() >= 2:
    cv=GroupKFold(n_splits=min(5,g.nunique()))
    for name,m in models.items():
        scores=cross_val_score(m,X,y,groups=g,cv=cv,scoring='roc_auc')
        results[name]={'mean_auc':float(scores.mean()),'std_auc':float(scores.std())}
(ROOT/'outputs'/'escalation_model_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
