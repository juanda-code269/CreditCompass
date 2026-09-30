import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="CreditScope", page_icon="🏦", layout="wide")
st.title("CreditScope — educational default-risk modeling")
st.warning("Demonstration only. Never use this app to approve, reject, price, or rank real applicants.")


@st.cache_data
def demo_data(n=3200):
    rng = np.random.default_rng(7)
    income = np.exp(rng.normal(10.8, .55, n)).clip(18000, 300000)
    loan = np.exp(rng.normal(10.15, .65, n)).clip(2000, 120000)
    age = rng.integers(21, 70, n)
    credit = np.clip(rng.normal(675, 65, n), 350, 850)
    debt_ratio = rng.beta(2.2, 4.5, n)
    employment = rng.choice(["salaried", "self-employed", "contract"], n, p=[.65,.2,.15])
    history = rng.choice(["thin", "established", "delinquent"], n, p=[.18,.7,.12])
    logit = -2.5 + 4*debt_ratio + 1.4*(loan/income) - .009*(credit-650) + .8*(history=="delinquent") + .35*(employment=="contract")
    p = 1/(1+np.exp(-logit)); default = rng.binomial(1, p)
    return pd.DataFrame({"income":income,"loan_amount":loan,"age":age,"credit_score":credit,
                         "debt_ratio":debt_ratio,"employment":employment,"credit_history":history,"default":default})


def load_data(upload):
    if upload is None:
        return demo_data(), "Clearly labeled synthetic demonstration data"
    df = pd.read_csv(upload)
    candidates = [c for c in df.columns if c.lower() in {"default","loan_status","target","bad_loan","label"}]
    if not candidates:
        raise ValueError("Could not identify the target. Name it default, loan_status, target, bad_loan, or label.")
    target = candidates[0]
    vals = df[target].dropna().unique()
    if len(vals) != 2:
        raise ValueError("The target must contain exactly two classes.")
    if not set(vals).issubset({0,1}):
        df[target] = (df[target] == vals[-1]).astype(int)
    return df.rename(columns={target:"default"}), f"Uploaded dataset: {upload.name}"


@st.cache_resource
def train_models(df):
    X, y = df.drop(columns="default"), df["default"].astype(int)
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), nums),
        ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats)
    ])
    tr, te = train_test_split(np.arange(len(df)), test_size=.25, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=260, min_samples_leaf=6, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    fitted, rows = {}, []
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)]).fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:,1]; pred = prob >= .5
        rows.append({"Model":name,"ROC-AUC":roc_auc_score(y.iloc[te],prob),"PR-AUC":average_precision_score(y.iloc[te],prob),
                     "Precision":precision_score(y.iloc[te],pred,zero_division=0),"Recall":recall_score(y.iloc[te],pred),"F1":f1_score(y.iloc[te],pred)})
        fitted[name]=(pipe, te, prob)
    return X, y, fitted, pd.DataFrame(rows), nums, cats


upload = st.sidebar.file_uploader("Optional borrower CSV", type="csv")
try:
    data, source = load_data(upload)
    if len(data) < 150: raise ValueError("At least 150 usable rows are required.")
    X, y, fitted, comparison, nums, cats = train_models(data.dropna(subset=["default"]))
except Exception as exc:
    st.error(str(exc)); st.stop()

model_name = st.sidebar.selectbox("Active model", comparison["Model"])
threshold = st.sidebar.slider("Decision threshold", .05, .95, .50, .01)
fp_cost = st.sidebar.number_input("Hypothetical rejection cost (false positive)", 0, 100000, 1000, 100)
fn_cost = st.sidebar.number_input("Hypothetical default cost (false negative)", 0, 100000, 12000, 500)
pipe, test_idx, prob = fitted[model_name]; truth = y.iloc[test_idx].to_numpy(); pred = prob >= threshold

tabs = st.tabs(["EDA", "Model comparison", "Threshold & cost", "Applicant scenario", "Explanations", "Responsibility"])
with tabs[0]:
    st.info(source)
    a,b,c = st.columns(3); a.metric("Rows", f"{len(data):,}"); b.metric("Default prevalence", f"{y.mean():.1%}"); c.metric("Missing cells", f"{data.isna().sum().sum():,}")
    c1,c2=st.columns(2)
    c1.plotly_chart(px.histogram(data, x="default", title="Class distribution"), width="stretch")
    if nums:
        feature= c2.selectbox("Numeric distribution", nums)
        c2.plotly_chart(px.histogram(data, x=feature, color=data["default"].astype(str), marginal="box"), width="stretch")
    st.dataframe(data.describe(include="all").T, width="stretch")

with tabs[1]:
    st.dataframe(comparison.style.format({c:"{:.3f}" for c in comparison.columns if c!="Model"}), width="stretch")
    fpr,tpr,_=roc_curve(truth,prob)
    st.plotly_chart(px.line(pd.DataFrame({"False-positive rate":fpr,"True-positive rate":tpr}),x="False-positive rate",y="True-positive rate",title=f"ROC — {model_name}"),width="stretch")
    frac, mean = calibration_curve(truth, prob, n_bins=10)
    cal = pd.DataFrame({"Mean predicted probability":mean,"Observed default rate":frac})
    st.plotly_chart(px.line(cal,x="Mean predicted probability",y="Observed default rate",markers=True,title="Calibration"),width="stretch")

with tabs[2]:
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    cols=st.columns(4)
    for col,label,val in zip(cols,["True negatives","False positives","False negatives","True positives"],[tn,fp,fn,tp]): col.metric(label,int(val))
    st.metric("Simulated expected cost", f"${fp*fp_cost + fn*fn_cost:,.0f}")
    sweep=[]
    for t in np.linspace(.02,.98,97):
        a,b,c,d=confusion_matrix(truth,prob>=t,labels=[0,1]).ravel()
        sweep.append({"Threshold":t,"False positives":b,"False negatives":c,"Cost":b*fp_cost+c*fn_cost})
    sw=pd.DataFrame(sweep); best=sw.loc[sw.Cost.idxmin()]
    st.success(f"Lowest simulated cost occurs near threshold {best.Threshold:.2f}. This depends entirely on the hypothetical costs.")
    st.plotly_chart(px.line(sw,x="Threshold",y=["False positives","False negatives"],title="Error trade-off"),width="stretch")
    st.plotly_chart(px.line(sw,x="Threshold",y="Cost",title="Simulated cost curve"),width="stretch")

with tabs[3]:
    row={}
    cols=st.columns(2)
    for i,col in enumerate(X.columns):
        s=X[col]
        with cols[i%2]:
            if col in nums:
                lo,hi=float(s.quantile(.02)),float(s.quantile(.98)); default=float(s.median())
                row[col]=st.number_input(col,lo,hi,default,key=f"input_{col}")
            else:
                row[col]=st.selectbox(col,sorted(s.dropna().astype(str).unique()),key=f"input_{col}")
    scenario=pd.DataFrame([row]); p=float(pipe.predict_proba(scenario)[0,1])
    label="High" if p>=.55 else "Moderate" if p>=.25 else "Low"
    st.metric("Model-estimated default probability",f"{p:.1%}",f"{label} model-defined band")
    st.caption("This estimate reflects the training data and model assumptions, not an objective truth about a person.")

with tabs[4]:
    prep=pipe.named_steps["prep"]; names=prep.get_feature_names_out(); model=pipe.named_steps["model"]
    importance=np.abs(model.coef_[0]) if hasattr(model,"coef_") else model.feature_importances_
    imp=pd.DataFrame({"Transformed feature":names,"Importance":importance}).nlargest(20,"Importance")
    st.plotly_chart(px.bar(imp.sort_values("Importance"),x="Importance",y="Transformed feature",orientation="h",title="Global model influence (magnitude)"),width="stretch")
    st.caption("Importance is association, not causation. For tree models it is not a signed contribution.")

with tabs[5]:
    st.markdown("""### Responsible-use notes
Credit models can reproduce historical inequity, measurement error, sampling bias, and label bias. Group-level performance should be audited when a legitimate governance process permits it; similar overall accuracy can conceal different false-positive or false-negative rates. A threshold embeds value judgments about who bears each error. This synthetic demonstration omits many real underwriting, legal, privacy, and compliance requirements and is **not production-ready**.""")

