# =============================================================
# eda_dashboard.py
# =============================================================
# Interactive Streamlit dashboard for exploring the Registan
# Chilonzor dropout dataset and presenting live predictions.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I built this as a Streamlit app:
#   My dissertation has two audiences — the academic committee and
#   Registan's operational team. A Streamlit app requires no frontend
#   infrastructure: a moderator with Python installed can run it with
#   a single command. This made it the right choice for a small
#   language school that has no dedicated web team.
#
# Why these seven pages:
#   Overview:             high-level KPIs and the monthly dropout trend
#                         — the first thing a manager wants to see.
#   Course Analysis:      I needed to verify that dropout rates differ
#                         meaningfully by course (they do — Korean is
#                         structurally harder to retain than General English).
#   Period Comparison:    lets Registan compare any two calendar periods
#                         side by side to detect seasonal patterns.
#   Teacher & Moderator:  the operational team asked specifically for
#                         per-teacher and per-moderator dropout rates.
#   Attendance & Payment: the two feature groups with the strongest
#                         predictive signal, shown as distributions.
#   Correlation & Features: academic transparency — I show the Pearson
#                           correlations used in my dissertation analysis.
#   Predictions:          the core deployment deliverable — a ranked,
#                         colour-coded list of at-risk active students
#                         that a moderator can filter and export.
#
# Input : data/processed/master_ml.parquet
#         data/processed/master_full.parquet
#         data/processed/snap_train.parquet
#         data/processed/predictions.parquet  (if predict.py has been run)
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.config import *

st.set_page_config(
    page_title="Registan Dropout Analysis",
    page_icon="Registan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Login screen ──────────────────────────────────────────────
# Credentials are stored in Streamlit Cloud secrets (Settings →
# Secrets) as:
#   [auth]
#   username = "your_username"
#   password = "your_password"
#
# Locally (without secrets) the dashboard is open by default.
def _check_login():
    try:
        creds = st.secrets["auth"]
        correct_user = creds["username"]
        correct_pass = creds["password"]
    except Exception:
        return True   # no secrets configured → open access (local dev)

    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🔒 Registan Dashboard</h2>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log in", use_container_width=True):
            if username == correct_user and password == correct_pass:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Wrong username or password.")
    st.stop()

_check_login()

@st.cache_data
def load_data():
    df   = pd.read_parquet(PROCESSED / "master_ml.parquet")
    full = pd.read_parquet(PROCESSED / "master_full.parquet")
    return df, full

@st.cache_data
def load_snap_train():
    return pd.read_parquet(PROCESSED / "snap_train.parquet")

df, full = load_data()
snap_global = load_snap_train()

# I decode the one-hot course, season, and shift columns into plain-text
# labels once here at load time. Every page then filters and labels charts
# by name rather than binary dummy columns, and Streamlit's caching means
# this transformation runs only once per session, not on every re-render.
_cc = [c for c in snap_global.columns if c.startswith('courseName_')]
_sc = [c for c in snap_global.columns if c.startswith('joinSeason_')]
_sh = [c for c in snap_global.columns if c.startswith('preferredShift_')]
snap_global = snap_global.copy()
snap_global['courseName']     = snap_global[_cc].idxmax(axis=1).str.replace('courseName_', '', regex=False)
snap_global['joinSeason']     = snap_global[_sc].idxmax(axis=1).str.replace('joinSeason_', '', regex=False)
snap_global['preferredShift'] = snap_global[_sh].idxmax(axis=1).str.replace('preferredShift_', '', regex=False)
snap_global['gender_label']   = snap_global['gender'].map({0: 'female', 1: 'male'})
snap_global['Status']         = snap_global['label'].map({1: 'Dropout', 0: 'Retained'})

# I reconstruct a proper datetime from the separate joinYear and joinMonth
# columns so the sidebar date slider can compare against real dates. The
# joinMonth column stores strings like "2022-09", so I extract the last
# two characters for the month number and pad to two digits before combining
# into a first-of-month date. tz_localize(None) strips UTC so Pandas does
# not raise a mixed-timezone error when comparing against the date inputs.
df['joinYear_str']  = df['joinYear'].fillna(2022).astype(int).astype(str)
df['joinMonth_str'] = df['joinMonth'].fillna('01')
df['joinMonth_num'] = df['joinMonth_str'].str[-2:].str.zfill(2)
df['joinDate'] = pd.to_datetime(
    df['joinYear_str'] + '-' + df['joinMonth_num'] + '-01',
    errors='coerce'
)
df['joinDate'] = df['joinDate'].dt.tz_localize(None)

LABEL_MAP = {1.0: 'Dropout', 0.0: 'Completer', -1.0: 'Active'}
df['label'] = df['dropout'].map(LABEL_MAP)

COLORS = {
    'Dropout'  : '#e74c3c',
    'Completer': '#2ecc71',
    'Active'   : '#3498db',
}
SNAP_COLORS = {
    'Dropout' : '#e74c3c',
    'Retained': '#2ecc71',
}

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.title("Registan EDA")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Course Analysis",
    "Period Comparison",
    "Teacher & Moderator",
    "Attendance & Payment",
    "Correlation & Features",
    "Predictions",
    "Live Predictions"
])

if page not in ("Predictions", "Live Predictions"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")

    # Reset button — sets all filters back to their default values
    def _reset_filters():
        st.session_state['f_course']  = 'All'
        st.session_state['f_gender']  = 'All'
        st.session_state['f_shift']   = 'All'
        st.session_state['f_season']  = 'All'
        st.session_state['f_alltime'] = True

    st.sidebar.button("↺ Reset all filters", on_click=_reset_filters)

    min_date = df['joinDate'].min()
    max_date = df['joinDate'].max()

    use_all_time = st.sidebar.checkbox("All time", value=True, key='f_alltime')
    if use_all_time:
        date_from = min_date
        date_to   = max_date
    else:
        date_from = pd.Timestamp(st.sidebar.date_input(
            "From", value=min_date, min_value=min_date, max_value=max_date
        ))
        date_to = pd.Timestamp(st.sidebar.date_input(
            "To", value=max_date, min_value=min_date, max_value=max_date
        ))

    all_courses = ['All'] + sorted(df['courseName'].dropna().unique().tolist())
    all_genders = ['All'] + sorted(df['gender'].dropna().unique().tolist())
    all_shifts  = ['All'] + sorted(df['preferredShift'].dropna().unique().tolist())
    all_seasons = ['All'] + sorted(df['joinSeason'].dropna().unique().tolist())

    sel_course = st.sidebar.selectbox("Course",  all_courses, key='f_course')
    sel_gender = st.sidebar.selectbox("Gender",  all_genders, key='f_gender')
    sel_shift  = st.sidebar.selectbox("Shift",   all_shifts,  key='f_shift')
    sel_season = st.sidebar.selectbox("Season",  all_seasons, key='f_season')

    fdf = df.copy()
    fdf = fdf[(fdf['joinDate'] >= date_from) & (fdf['joinDate'] <= date_to)]
    if sel_course != 'All': fdf = fdf[fdf['courseName']     == sel_course]
    if sel_gender != 'All': fdf = fdf[fdf['gender']         == sel_gender]
    if sel_shift  != 'All': fdf = fdf[fdf['preferredShift'] == sel_shift]
    if sel_season != 'All': fdf = fdf[fdf['joinSeason']     == sel_season]

    train = fdf[fdf['dropout'].isin([0, 1])].copy()

    snap_fdf = snap_global.copy()
    if sel_course != 'All': snap_fdf = snap_fdf[snap_fdf['courseName']     == sel_course]
    if sel_gender != 'All': snap_fdf = snap_fdf[snap_fdf['gender_label']   == sel_gender]
    if sel_shift  != 'All': snap_fdf = snap_fdf[snap_fdf['preferredShift'] == sel_shift]
    if sel_season != 'All': snap_fdf = snap_fdf[snap_fdf['joinSeason']     == sel_season]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Snap rows after filters: {len(snap_fdf):,}")

else:
    # Predictions / Live Predictions — no global filters shown, set defaults
    min_date   = df['joinDate'].min()
    max_date   = df['joinDate'].max()
    date_from  = min_date
    date_to    = max_date
    sel_course = 'All'
    sel_gender = 'All'
    sel_shift  = 'All'
    sel_season = 'All'
    fdf        = df.copy()
    train      = fdf[fdf['dropout'].isin([0, 1])].copy()
    snap_fdf   = snap_global.copy()

# =============================================================
# OVERVIEW PAGE
# =============================================================
if page == "Overview":
    st.title("Overview Dashboard")
    st.markdown("---")

    # Use snap_fdf which already has all global filters applied (course, gender, shift, season)
    snap_f = snap_fdf.copy()

    n_labeled_students = snap_f['studentId'].nunique()
    n_dropout_snap     = int((snap_f['label'] == 1).sum())
    n_retained_snap    = int((snap_f['label'] == 0).sum())
    n_active           = int((df['dropout'] == -1).sum())
    total              = n_labeled_students + n_active
    dropout_rate       = n_dropout_snap / len(snap_f) * 100 if len(snap_f) > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",        f"{total:,}")
    c2.metric("Dropout Snapshots",     f"{n_dropout_snap:,}")
    c3.metric("Retained Snapshots",    f"{n_retained_snap:,}")
    c4.metric("Active (scoring)",      f"{n_active:,}")
    c5.metric("Snapshot Dropout Rate", f"{dropout_rate:.1f}%")

    st.caption(
        "Metrics based on the snapshot training set — one row per (student, month). "
        "Label = 1 means the student had no lessons in the following 30 days (dropout signal). "
        "Active students are the current scoring cohort excluded from training."
    )

    st.markdown("---")

    # snap_f already has courseName decoded (added at load time via snap_global)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Label Distribution")
        label_dist = pd.DataFrame({
            'Status': ['Dropout (label=1)', 'Retained (label=0)', 'Active (scoring)'],
            'Count':  [n_dropout_snap, n_retained_snap, n_active]
        })
        fig = px.pie(
            label_dist, names='Status', values='Count',
            color='Status',
            color_discrete_map={
                'Dropout (label=1)' : '#e74c3c',
                'Retained (label=0)': '#2ecc71',
                'Active (scoring)'  : '#3498db',
            },
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Dropout Rate by Course")
        course_stats = snap_f.groupby('courseName').agg(
            total    = ('label', 'count'),
            dropouts = ('label', 'sum')
        ).reset_index()
        course_stats['dropout_rate'] = (
            course_stats['dropouts'] / course_stats['total'] * 100
        ).round(1)
        course_stats = course_stats.sort_values('dropout_rate', ascending=True)
        fig = px.bar(
            course_stats, x='dropout_rate', y='courseName',
            orientation='h',
            color='dropout_rate',
            color_continuous_scale='RdYlGn_r',
            text='dropout_rate',
            labels={'dropout_rate': 'Dropout Rate (%)', 'courseName': 'Course'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Dropout Rate Trend")
    st.caption("% of snapshots labelled dropout=1 in each month (30-day forward attendance window).")

    monthly = (
        snap_f.groupby('snapshotMonth')['label']
        .agg(['sum', 'count'])
        .reset_index()
    )
    monthly['dropout_rate'] = (monthly['sum'] / monthly['count'] * 100).round(1)
    monthly = monthly.sort_values('snapshotMonth')

    fig = px.line(
        monthly, x='snapshotMonth', y='dropout_rate',
        markers=True,
        labels={'snapshotMonth': 'Month', 'dropout_rate': 'Dropout Rate (%)'},
    )
    fig.update_traces(line_color='#e74c3c', marker_color='#e74c3c')
    fig.add_hline(
        y=dropout_rate, line_dash='dash', line_color='gray',
        annotation_text=f'Overall avg: {dropout_rate:.1f}%'
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================
# COURSE ANALYSIS PAGE
# =============================================================
elif page == "Course Analysis":
    st.title("Course Analysis")
    st.markdown("---")

    course_choice = st.selectbox(
        "Select course to analyse",
        ['All courses'] + sorted(snap_global['courseName'].unique().tolist())
    )

    cdf = snap_fdf.copy()
    if course_choice != 'All courses':
        cdf = cdf[cdf['courseName'] == course_choice]

    c1, c2, c3 = st.columns(3)
    n_snap   = len(cdf)
    n_drop   = int((cdf['label'] == 1).sum())
    dr       = n_drop / n_snap * 100 if n_snap > 0 else 0
    c1.metric("Snapshots",             f"{n_snap:,}")
    c2.metric("Dropout Snapshots",     f"{n_drop:,}")
    c3.metric("Snapshot Dropout Rate", f"{dr:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Overall Attendance Rate by Course")
        att_by_course = snap_fdf.groupby(['courseName', 'Status'])['attendanceRate'].mean().reset_index()
        fig = px.bar(
            att_by_course, x='courseName', y='attendanceRate',
            color='Status', color_discrete_map=SNAP_COLORS, barmode='group',
            labels={'attendanceRate': 'Avg Attendance Rate', 'courseName': 'Course'}
        )
        fig.update_yaxes(tickformat='.0%')
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Last 30-Day Attendance by Course")
        att30_by_course = snap_fdf.groupby(['courseName', 'Status'])['attendanceLast30Days'].mean().reset_index()
        fig = px.bar(
            att30_by_course, x='courseName', y='attendanceLast30Days',
            color='Status', color_discrete_map=SNAP_COLORS, barmode='group',
            labels={'attendanceLast30Days': 'Avg Last-30-Day Attendance', 'courseName': 'Course'}
        )
        fig.update_yaxes(tickformat='.0%')
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Dropout Rate by Shift")
        shift_stats = cdf.groupby('preferredShift').agg(
            total    = ('label', 'count'),
            dropouts = ('label', 'sum')
        ).reset_index()
        shift_stats['dropout_rate'] = (shift_stats['dropouts'] / shift_stats['total'] * 100).round(1)
        fig = px.bar(
            shift_stats, x='preferredShift', y='dropout_rate',
            color='dropout_rate', color_continuous_scale='RdYlGn_r',
            text='dropout_rate',
            labels={'dropout_rate': 'Dropout Rate (%)', 'preferredShift': 'Shift'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Dropout Rate by Season")
        season_stats = cdf.groupby('joinSeason').agg(
            total    = ('label', 'count'),
            dropouts = ('label', 'sum')
        ).reset_index()
        season_stats['dropout_rate'] = (season_stats['dropouts'] / season_stats['total'] * 100).round(1)
        fig = px.bar(
            season_stats, x='joinSeason', y='dropout_rate',
            color='dropout_rate', color_continuous_scale='RdYlGn_r',
            text='dropout_rate',
            labels={'dropout_rate': 'Dropout Rate (%)', 'joinSeason': 'Season'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# PERIOD COMPARISON PAGE
# =============================================================
elif page == "Period Comparison":
    st.title("Period Comparison")
    st.markdown("Compare two snapshot periods side by side.")
    st.markdown("---")

    snap_months = sorted(snap_fdf['snapshotMonth'].unique().tolist())
    mid = len(snap_months) // 2

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Period A")
        pa_from = st.selectbox("From month", snap_months, index=0, key='pa_from')
        pa_to   = st.selectbox("To month",   snap_months, index=max(mid - 1, 0), key='pa_to')
    with col2:
        st.subheader("Period B")
        pb_from = st.selectbox("From month", snap_months, index=mid, key='pb_from')
        pb_to   = st.selectbox("To month",   snap_months, index=len(snap_months) - 1, key='pb_to')

    pa = snap_fdf[(snap_fdf['snapshotMonth'] >= pa_from) & (snap_fdf['snapshotMonth'] <= pa_to)]
    pb = snap_fdf[(snap_fdf['snapshotMonth'] >= pb_from) & (snap_fdf['snapshotMonth'] <= pb_to)]

    st.markdown("---")

    col_a, col_b = st.columns(2)

    def show_period_metrics(pdata, label):
        n      = len(pdata)
        n_drop = int((pdata['label'] == 1).sum())
        dr     = n_drop / n * 100 if n > 0 else 0
        st.metric(f"{label} — Snapshots",     f"{n:,}")
        st.metric(f"{label} — Dropout Rate",  f"{dr:.1f}%")
        st.metric(f"{label} — Avg Attendance",f"{pdata['attendanceRate'].mean():.1%}")
        st.metric(f"{label} — Avg Pay Months",f"{pdata['paymentMonths'].mean():.1f}")

    with col_a:
        show_period_metrics(pa, "Period A")
    with col_b:
        show_period_metrics(pb, "Period B")

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Dropout Rate by Course")
        def course_dr(data, name):
            s = data.groupby('courseName').agg(
                total    = ('label', 'count'),
                dropouts = ('label', 'sum')
            ).reset_index()
            s['dropout_rate'] = s['dropouts'] / s['total'] * 100
            s['period'] = name
            return s
        comp = pd.concat([course_dr(pa, 'Period A'), course_dr(pb, 'Period B')])
        fig = px.bar(
            comp, x='courseName', y='dropout_rate',
            color='period', barmode='group',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            labels={'dropout_rate': 'Dropout Rate (%)', 'courseName': 'Course'}
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Attendance Rate Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pa['attendanceRate'], name='Period A',
            opacity=0.6, marker_color='#3498db', nbinsx=20
        ))
        fig.add_trace(go.Histogram(
            x=pb['attendanceRate'], name='Period B',
            opacity=0.6, marker_color='#e74c3c', nbinsx=20
        ))
        fig.update_layout(
            barmode='overlay',
            xaxis_title='Attendance Rate',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Payment Months Distribution")
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=pa['paymentMonths'], name='Period A', marker_color='#3498db'
        ))
        fig.add_trace(go.Box(
            y=pb['paymentMonths'], name='Period B', marker_color='#e74c3c'
        ))
        fig.update_layout(yaxis_title='Payment Months')
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.subheader("Debt Rate Comparison")
        debt_comp = pd.DataFrame({
            'Period'     : ['Period A', 'Period B'],
            'Debt Rate'  : [pa['debtRate'].mean()*100, pb['debtRate'].mean()*100],
            'Unpaid Rate': [pa['unpaidRate'].mean()*100, pb['unpaidRate'].mean()*100]
        })
        fig = px.bar(
            debt_comp.melt(id_vars='Period'),
            x='variable', y='value', color='Period',
            barmode='group',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            labels={'value': 'Rate (%)', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# TEACHER & MODERATOR PAGE
# =============================================================
elif page == "Teacher & Moderator":
    st.title("Teacher & Moderator Analysis")
    st.markdown("---")
    st.caption(
        "Teacher and moderator names are sourced from administrative records (master_full.parquet). "
        "Dropout rates here reflect actual student outcomes rather than the 30-day snapshot label "
        "used for model training — this is intentional since teacher performance is evaluated on "
        "whether students ultimately stayed or left, not on a monthly attendance window."
    )

    # I re-apply the global filters to master_full rather than reusing fdf
    # because the Teacher & Moderator page needs identifier columns like
    # lastTeacherName and lastModeratorName that only exist in master_full.
    # These were deliberately excluded from master_ml to avoid leaking
    # non-predictive identifiers into the model.
    full_f = full.copy()
    full_f['joinYear_str']  = full_f['joinYear'].fillna(2022).astype(int).astype(str)
    full_f['joinMonth_str'] = full_f['joinMonth'].fillna('01')
    full_f['joinMonth_num'] = full_f['joinMonth_str'].str[-2:].str.zfill(2)
    full_f['joinDate'] = pd.to_datetime(
        full_f['joinYear_str'] + '-' + full_f['joinMonth_num'] + '-01',
        errors='coerce'
    ).dt.tz_localize(None)
    full_f = full_f[
        (full_f['joinDate'] >= date_from) &
        (full_f['joinDate'] <= date_to)
    ]
    if sel_course != 'All': full_f = full_f[full_f['courseName'] == sel_course]
    if sel_gender != 'All': full_f = full_f[full_f['gender'] == sel_gender]
    if sel_shift  != 'All': full_f = full_f[full_f['preferredShift'] == sel_shift]
    if sel_season != 'All': full_f = full_f[full_f['joinSeason'] == sel_season]

    full_train = full_f[full_f['dropout'].isin([0, 1])].copy()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Teacher Dropout Rate (top 20)")
        teacher_agg = full_train.groupby(
            ['lastTeacherId', 'lastTeacherName']
        ).agg(
            total    = ('dropout', 'count'),
            dropouts = ('dropout', 'sum'),
        ).reset_index()
        teacher_agg['dropout_rate'] = (
            teacher_agg['dropouts'] / teacher_agg['total'] * 100
        ).round(1)
        teacher_agg = teacher_agg[
            teacher_agg['total'] >= 10
        ].sort_values('dropout_rate', ascending=False).head(20)
        teacher_agg = teacher_agg.sort_values('dropout_rate', ascending=True)

        fig = px.bar(
            teacher_agg,
            x='dropout_rate', y='lastTeacherName',
            orientation='h',
            color='dropout_rate',
            color_continuous_scale='RdYlGn_r',
            text='dropout_rate',
            hover_data={'total': True, 'lastTeacherId': False, 'lastTeacherName': False},
            labels={
                'dropout_rate'    : 'Dropout Rate (%)',
                'lastTeacherName' : 'Teacher',
                'total'           : 'Total Students',
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=550)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Teacher Avg Attendance Rate")
        st.caption("Average student attendance rate per teacher. Min 10 students.")
        teacher_att = full_train.groupby(
            ['lastTeacherId', 'lastTeacherName']
        ).agg(
            total   = ('dropout', 'count'),
            avg_att = ('lastTeacherAvgAttendanceRate', 'mean'),
            trend   = ('lastTeacherAttendanceTrend', 'mean'),
            dr      = ('dropout', 'mean')
        ).reset_index()
        teacher_att = teacher_att[
            teacher_att['total'] >= 10
        ].sort_values('avg_att', ascending=True)
        teacher_att['avg_att_pct'] = (teacher_att['avg_att'] * 100).round(1)
        teacher_att['trend_label'] = teacher_att['trend'].apply(
    lambda x: f'Improving (+{x:.3f})' if x > 0.01 else (
        f'Declining ({x:.3f})' if x < -0.01 else f'Stable ({x:.3f})'
    )
)

        fig = px.bar(
            teacher_att,
            x='avg_att_pct', y='lastTeacherName',
            orientation='h',
            color='trend_label',
            color_discrete_map={
                'Improving': '#2ecc71',
                'Stable'   : '#f39c12',
                'Declining': '#e74c3c'
            },
            text='avg_att_pct',
            hover_data={'total': True, 'avg_att_pct': False},
            labels={
                'avg_att_pct'     : 'Avg Attendance Rate (%)',
                'lastTeacherName' : 'Teacher',
                'trend_label'     : 'Trend',
                'total'           : 'Students'
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Moderator Dropout Rate")

    mod_agg = full_train.groupby(
        ['lastModeratorId', 'lastModeratorName']
    ).agg(
        total    = ('dropout', 'count'),
        dropouts = ('dropout', 'sum')
    ).reset_index()
    mod_agg['dropout_rate'] = (
        mod_agg['dropouts'] / mod_agg['total'] * 100
    ).round(1)
    mod_agg = mod_agg[
        mod_agg['total'] >= 20
    ].sort_values('dropout_rate', ascending=False).reset_index(drop=True)

    fig = px.bar(
        mod_agg,
        x='lastModeratorName', y='dropout_rate',
        color='dropout_rate',
        color_continuous_scale='RdYlGn_r',
        text='dropout_rate',
        hover_data={'lastModeratorId': False, 'total': True, 'lastModeratorName': False},
        labels={
            'dropout_rate'      : 'Dropout Rate (%)',
            'lastModeratorName' : 'Moderator',
            'total'             : 'Total Students'
        }
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

  # =============================================================
# ATTENDANCE & PAYMENT PAGE
# =============================================================
elif page == "Attendance & Payment":
    st.title("Attendance & Payment Analysis")
    st.markdown("---")

    dropout_data  = snap_fdf[snap_fdf['label'] == 1]
    retained_data = snap_fdf[snap_fdf['label'] == 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dropout Avg Att. (30d)",   f"{dropout_data['attendanceLast30Days'].mean():.1%}")
    c2.metric("Retained Avg Att. (30d)",  f"{retained_data['attendanceLast30Days'].mean():.1%}")
    c3.metric("Dropout Avg Pay Months",   f"{dropout_data['paymentMonths'].mean():.1f}")
    c4.metric("Retained Avg Pay Months",  f"{retained_data['paymentMonths'].mean():.1f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Last-30-Day Attendance Distribution")
        st.caption("Attendance rate in the 30 days before each snapshot — the model's strongest predictor.")
        fig = px.histogram(
            snap_fdf, x='attendanceLast30Days',
            color='Status', color_discrete_map=SNAP_COLORS,
            nbins=20, opacity=0.75, barmode='overlay',
            labels={'attendanceLast30Days': 'Attendance Rate (last 30 days)', 'Status': 'Label'}
        )
        fig.update_xaxes(tickformat='.0%')
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            yaxis_title='Number of Snapshots'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Overall Attendance Rate Distribution")
        st.caption("Cumulative attendance rate across the full enrolment period.")
        fig = px.histogram(
            snap_fdf, x='attendanceRate',
            color='Status', color_discrete_map=SNAP_COLORS,
            nbins=20, opacity=0.75, barmode='overlay',
            labels={'attendanceRate': 'Overall Attendance Rate', 'Status': 'Label'}
        )
        fig.update_xaxes(tickformat='.0%')
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            yaxis_title='Number of Snapshots'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Payment Months Distribution")
        st.caption("How many months have dropout vs retained students paid?")
        fig = px.histogram(
            snap_fdf, x='paymentMonths',
            color='Status', color_discrete_map=SNAP_COLORS,
            nbins=20, opacity=0.75, barmode='overlay',
            labels={'paymentMonths': 'Payment Months', 'Status': 'Label'}
        )
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            yaxis_title='Number of Snapshots'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Average Debt & Unpaid Rate by Label")
        st.caption("Proportion of months ending in debt or with unpaid balance.")
        debt_summary = snap_fdf.groupby('Status').agg(
            avg_debt   = ('debtRate', 'mean'),
            avg_unpaid = ('unpaidRate', 'mean')
        ).reset_index()
        debt_summary['avg_debt_pct']   = (debt_summary['avg_debt'] * 100).round(1)
        debt_summary['avg_unpaid_pct'] = (debt_summary['avg_unpaid'] * 100).round(1)
        fig = px.bar(
            debt_summary.melt(id_vars='Status',
                              value_vars=['avg_debt_pct', 'avg_unpaid_pct'],
                              var_name='metric', value_name='value'),
            x='Status', y='value', color='metric', barmode='group',
            color_discrete_map={'avg_debt_pct': '#e74c3c', 'avg_unpaid_pct': '#e67e22'},
            text='value',
            labels={'value': 'Rate (%)', 'Status': 'Label', 'metric': 'Metric'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.for_each_trace(lambda t: t.update(
            name='Debt Rate' if 'debt' in t.name else 'Unpaid Rate'
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Attendance Trend by Label")
        st.caption("Green = improving. Red = declining over time.")
        trend_summary = snap_fdf.groupby('Status')['attendanceTrend'].apply(
            lambda x: pd.Series({
                'Improving (>0)' : (x > 0.01).sum(),
                'Stable'         : ((x >= -0.01) & (x <= 0.01)).sum(),
                'Declining (<0)' : (x < -0.01).sum()
            })
        ).reset_index()
        trend_summary.columns = ['Status', 'trend_type', 'count']
        fig = px.bar(
            trend_summary, x='Status', y='count', color='trend_type', barmode='group',
            color_discrete_map={
                'Improving (>0)': '#2ecc71',
                'Stable'        : '#f39c12',
                'Declining (<0)': '#e74c3c'
            },
            text='count',
            labels={'count': 'Number of Snapshots', 'Status': 'Label', 'trend_type': 'Trend'}
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.subheader("Consecutive Missed Lessons")
        st.caption("Average streak of missed lessons at snapshot time.")
        missed_summary = snap_fdf.groupby('Status').agg(
            avg_missed = ('consecutiveMissedLessons', 'mean'),
        ).reset_index()
        missed_summary['avg_missed'] = missed_summary['avg_missed'].round(1)
        fig = px.bar(
            missed_summary, x='Status', y='avg_missed',
            color='Status', color_discrete_map=SNAP_COLORS,
            text='avg_missed',
            labels={'avg_missed': 'Avg Consecutive Missed', 'Status': 'Label'}
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# CORRELATION & FEATURES PAGE
# =============================================================
elif page == "Correlation & Features":
    st.title("Correlation & Feature Analysis")
    st.markdown("---")

    FIGURES = Path("reports/figures")

    snap = snap_fdf  # respects all global filters

    EXCLUDE_COLS = ['studentId', 'snapshotMonth', 'label', 'rowSet']
    num_cols = [
        c for c in snap.columns
        if c not in EXCLUDE_COLS
        and snap[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Feature Correlation with Dropout")
        st.caption("Red = positively correlated with dropout. Green = negatively correlated.")
        corr_img = FIGURES / "target_correlation.png"
        if corr_img.exists():
            st.image(str(corr_img), use_container_width=True)
        else:
            st.warning("Run src/analysis/correlation.py first.")

    with col_b:
        st.subheader("Top 20 Features — Live Ranking")
        st.caption("Computed from snapshot training data (48,448 monthly observations).")
        target_corr = snap[num_cols + ['label']].corr()['label'].drop('label')
        target_corr_sorted = target_corr.reindex(
            target_corr.abs().sort_values(ascending=False).index
        ).head(20)
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in target_corr_sorted.values]
        fig = go.Figure(go.Bar(
            x=target_corr_sorted.values,
            y=target_corr_sorted.index,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in target_corr_sorted.values],
            textposition='outside'
        ))
        fig.add_vline(x=0, line_color='white', line_width=1)
        fig.update_layout(
            xaxis_title='Pearson r with dropout label',
            yaxis=dict(autorange='reversed'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Correlation Heatmap")
    st.caption("Lower triangle only. Red = positive correlation. Blue = negative.")
    heatmap_img = FIGURES / "correlation_matrix.png"
    if heatmap_img.exists():
        st.image(str(heatmap_img), use_container_width=True)
    else:
        st.warning("Run src/analysis/correlation.py first.")

    st.markdown("---")
    st.subheader("Feature Averages by Label")
    st.caption("Compare average feature values between dropouts and completers.")

    LABEL_MAP_SNAP = {1: 'Dropout', 0: 'Retained'}
    snap_display = snap.copy()
    snap_display['Status'] = snap_display['label'].map(LABEL_MAP_SNAP)

    show_cols = st.multiselect(
        "Select features to compare",
        options=num_cols,
        default=[
            c for c in ['attendanceLast30Days', 'attendanceLast60Days',
                        'attendanceRate', 'currentBalance',
                        'paymentMonths', 'debtRate', 'consecutiveMissedLessons']
            if c in num_cols
        ]
    )
    if show_cols:
        stats = snap_display.groupby('Status')[show_cols].mean().T.round(3)

        # I normalise each row to 0–1 relative to its own maximum so features
        # on completely different scales — currentBalance measured in UZS versus
        # attendanceRate between 0 and 1 — can appear side by side without one
        # bar dwarfing all the others. I still print the actual mean values as
        # bar labels so the chart remains quantitatively honest.
        stats_norm = stats.copy()
        for feat in stats_norm.index:
            row_max = stats_norm.loc[feat].abs().max()
            if row_max > 0:
                stats_norm.loc[feat] = stats_norm.loc[feat] / row_max

        fig_avg = go.Figure()
        if 'Dropout' in stats_norm.columns:
            fig_avg.add_trace(go.Bar(
                name='Dropout',
                x=stats_norm.index,
                y=stats_norm['Dropout'],
                marker_color='#e74c3c',
                text=[f"{v:.3f}" for v in stats['Dropout']],
                textposition='outside'
            ))
        if 'Retained' in stats_norm.columns:
            fig_avg.add_trace(go.Bar(
                name='Retained',
                x=stats_norm.index,
                y=stats_norm['Retained'],
                marker_color='#2ecc71',
                text=[f"{v:.3f}" for v in stats['Retained']],
                textposition='outside'
            ))
        fig_avg.update_layout(
            barmode='group',
            height=450,
            xaxis_title='Feature',
            yaxis=dict(title='Relative magnitude (normalised per feature)',
                       range=[0, 1.35]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(t=40, b=80)
        )
        st.plotly_chart(fig_avg, use_container_width=True)
        st.caption("Bar heights are normalised per feature so different scales are "
                   "comparable. Actual mean values shown as labels above each bar.")
        st.dataframe(stats, use_container_width=True)

# =============================================================
# PREDICTIONS PAGE  (7th page — the moderator's one-click tool)
# =============================================================
# I designed this page as the core operational deliverable for
# Registan's moderators. It loads the output written by predict.py
# and presents each currently-active student's dropout probability
# alongside top risk factors in plain language. A moderator can
# filter to their own student list, sort by risk level, and download
# the table as CSV — all without any knowledge of machine learning.
# The colour-coded risk column (red/amber/green) lets them scan the
# table in seconds and decide who to call that day.
elif page == "Predictions":
    st.title("Dropout Predictions — Active Students")
    st.caption("Probability each currently-active student stops attending next month, "
               "with the top risk factors driving each score.")

    # ---------------------------------------------------------
    # load predictions + attach human-readable names
    # ---------------------------------------------------------
    @st.cache_data
    def load_predictions():
        pred_path = PROCESSED / "predictions.parquet"
        if not pred_path.exists():
            return None
        pred = pd.read_parquet(pred_path)
        # Attach student names from student_names.parquet (names only, no phones).
        # Falls back to full students.parquet locally, or studentId on cloud.
        names_path = PROCESSED / "student_names.parquet"
        if names_path.exists():
            stu = pd.read_parquet(names_path)
            pred = pred.merge(
                stu[['studentId', 'fullName']].rename(columns={'fullName': 'studentName'}),
                on='studentId', how='left'
            )
        elif Path(STUDENTS_CLEAN).exists():
            stu = pd.read_parquet(STUDENTS_CLEAN)
            pred = pred.merge(
                stu[['studentId', 'fullName']].rename(columns={'fullName': 'studentName'}),
                on='studentId', how='left'
            )
        else:
            pred['studentName'] = pred['studentId']
        if 'phoneNumber' not in pred.columns:
            pred['phoneNumber'] = ''
        # I pull teacher and moderator names from master_full because the
        # predictions file only stores IDs. The moderator needs names to know
        # which colleague to contact if a student is flagged as high risk.
        ctx = full[['studentId', 'courseName', 'lastTeacherName', 'lastModeratorName']].drop_duplicates(
            ['studentId', 'courseName']
        )
        pred = pred.merge(ctx, on=['studentId', 'courseName'], how='left')
        pred['studentName']      = pred['studentName'].fillna('Unknown')
        pred['lastTeacherName']  = pred['lastTeacherName'].fillna('Unknown')
        pred['lastModeratorName'] = pred['lastModeratorName'].fillna('Unknown')
        return pred

    pred = load_predictions()

    if pred is None:
        st.warning(
            "No predictions found. Run the model pipeline first:\n\n"
            "`python3 src/models/train_models.py`  then  `python3 src/models/predict.py`"
        )
    else:
        # ---- summary metrics ---------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active students", f"{len(pred):,}")
        c2.metric("High risk (>70%)",  int((pred['risk_level'] == 'High').sum()))
        c3.metric("Medium risk (40-70%)", int((pred['risk_level'] == 'Medium').sum()))
        c4.metric("Low risk (<40%)",   int((pred['risk_level'] == 'Low').sum()))

        st.markdown("---")

        # ---- reset button ------------------------------------
        def _reset_pred_filters():
            for k in ['p_risk', 'p_course', 'p_mod']:
                st.session_state[k] = 'All'
            for k in ['p_name', 'p_phone']:
                st.session_state[k] = None

        st.button("↺ Reset filters", on_click=_reset_pred_filters, key='pred_reset')

        # ---- filters -----------------------------------------
        f1, f2, f3 = st.columns(3)
        risk_opts   = ['All', 'High', 'Medium', 'Low']
        course_opts = ['All'] + sorted(pred['courseName'].dropna().unique().tolist())
        mod_opts    = ['All'] + sorted(pred['lastModeratorName'].dropna().unique().tolist())

        sel_risk    = f1.selectbox("Risk level",              risk_opts,   key='p_risk')
        sel_pcourse = f2.selectbox("Course",                  course_opts, key='p_course')
        sel_mod     = f3.selectbox("My students (moderator)", mod_opts,    key='p_mod')

        s1, s2 = st.columns(2)
        name_opts  = sorted(pred['studentName'].dropna().unique().tolist())
        phone_opts = sorted(pred['phoneNumber'].dropna().astype(str).unique().tolist())
        search_name  = s1.selectbox("Search by student name",  name_opts,  key='p_name',  index=None, placeholder="Type a name...")
        search_phone = s2.selectbox("Search by phone number",  phone_opts, key='p_phone', index=None, placeholder="Type a phone number...")

        view = pred.copy()
        if sel_risk    != 'All':  view = view[view['risk_level']        == sel_risk]
        if sel_pcourse != 'All':  view = view[view['courseName']        == sel_pcourse]
        if sel_mod     != 'All':  view = view[view['lastModeratorName'] == sel_mod]
        if search_name  is not None: view = view[view['studentName']               == search_name]
        if search_phone is not None: view = view[view['phoneNumber'].astype(str)   == search_phone]

        st.caption(f"Showing {len(view):,} of {len(pred):,} active students")

        # ---- risk distribution by course ---------------------
        if len(view) > 0:
            dist = view.groupby(['courseName', 'risk_level']).size().reset_index(name='n')
            fig = px.bar(
                dist, x='courseName', y='n', color='risk_level',
                color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'},
                title="Risk distribution by course", labels={'n': 'Students', 'courseName': 'Course'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- ranked, colour-coded table ----------------------
        table = view.sort_values('dropout_probability', ascending=False)[[
            'studentName', 'phoneNumber', 'courseName', 'dropout_probability', 'risk_level',
            'topFactors', 'lastTeacherName', 'lastModeratorName'
        ]].rename(columns={
            'studentName': 'Student', 'phoneNumber': 'Phone',
            'courseName': 'Course',
            'dropout_probability': 'Risk %', 'risk_level': 'Risk',
            'topFactors': 'Top Factors',
            'lastTeacherName': 'Teacher', 'lastModeratorName': 'Moderator'
        }).reset_index(drop=True)
        table['Risk %'] = (table['Risk %'] * 100).round(1)

        def colour_risk(val):
            return {
                'High':   'background-color: #e74c3c; color: white',
                'Medium': 'background-color: #f39c12; color: white',
                'Low':    'background-color: #2ecc71; color: white',
            }.get(val, '')

        st.dataframe(
            table.style.map(colour_risk, subset=['Risk'])
                       .format({'Risk %': '{:.1f}'}),
            use_container_width=True, height=480
        )

        # ---- CSV export for moderators -----------------------
        st.download_button(
            "Download list as CSV",
            data=table.to_csv(index=False).encode('utf-8'),
            file_name="dropout_predictions.csv",
            mime="text/csv"
        )

# =============================================================
# LIVE PREDICTIONS PAGE
# =============================================================
elif page == "Live Predictions":
    st.title("Live Predictions — Active Students")
    st.caption(
        "Dropout risk scores generated from **freshly pulled API data** (from 2026-03-19 onwards). "
        "Run `python3 api_on_process/run_live_pipeline.py` to refresh."
    )

    live_path = PROCESSED / "live_predictions.parquet"

    if not live_path.exists():
        st.warning(
            "No live predictions found yet.\n\n"
            "To generate them, run in your terminal:\n\n"
            "```\ncd dissertation_final\n"
            "python3 api_on_process/run_live_pipeline.py\n```\n\n"
            "This fetches fresh data from the API, cleans it, runs feature engineering, "
            "and scores active students with the trained model."
        )
    else:
        @st.cache_data
        def load_live_predictions():
            pred = pd.read_parquet(live_path)

            # Names are pre-enriched by predict_live.py (live_interim + original interim).
            # If the file was generated by an older pipeline run without enrichment,
            # fall back to merging from original interim + master_full as before.
            if 'studentName' not in pred.columns:
                names_path = PROCESSED / "student_names.parquet"
                if names_path.exists():
                    stu = pd.read_parquet(names_path)
                    pred = pred.merge(
                        stu[['studentId', 'fullName']].rename(columns={'fullName': 'studentName'}),
                        on='studentId', how='left'
                    )
                elif Path(STUDENTS_CLEAN).exists():
                    stu = pd.read_parquet(STUDENTS_CLEAN)
                    pred = pred.merge(
                        stu[['studentId', 'fullName']].rename(columns={'fullName': 'studentName'}),
                        on='studentId', how='left'
                    )
                else:
                    pred['studentName'] = pred['studentId']
                ctx = full[['studentId', 'courseName', 'lastTeacherName', 'lastModeratorName']].drop_duplicates(
                    ['studentId', 'courseName']
                )
                pred = pred.merge(ctx, on=['studentId', 'courseName'], how='left')

            # ensure all display columns exist and have no nulls
            pred['studentName']       = pred.get('studentName',       pd.Series('Unknown', index=pred.index)).fillna('Unknown')
            pred['lastTeacherName']   = pred.get('lastTeacherName',   pd.Series('Unknown', index=pred.index)).fillna('Unknown')
            pred['lastModeratorName'] = pred.get('lastModeratorName', pd.Series('Unknown', index=pred.index)).fillna('Unknown')
            if 'phoneNumber' not in pred.columns:
                pred['phoneNumber'] = ''
            pred['phoneNumber'] = pred['phoneNumber'].fillna('').astype(str).replace('None', '')
            return pred

        live = load_live_predictions()

        # ---- retrieved timestamp ----------------------------
        if 'retrievedAt' in live.columns:
            retrieved = live['retrievedAt'].iloc[0]
            st.info(f"Data retrieved at: **{retrieved}**")

        # ---- summary metrics --------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active students",       f"{len(live):,}")
        c2.metric("High risk (>70%)",      int((live['risk_level'] == 'High').sum()))
        c3.metric("Medium risk (40-70%)",  int((live['risk_level'] == 'Medium').sum()))
        c4.metric("Low risk (<40%)",       int((live['risk_level'] == 'Low').sum()))

        st.markdown("---")

        # ---- reset button -----------------------------------
        def _reset_live_filters():
            for k in ['lv_risk', 'lv_course', 'lv_mod']:
                st.session_state[k] = 'All'
            for k in ['lv_name', 'lv_phone']:
                st.session_state[k] = None

        st.button("↺ Reset filters", on_click=_reset_live_filters, key='live_reset')

        # ---- filters ----------------------------------------
        f1, f2, f3 = st.columns(3)
        risk_opts   = ['All', 'High', 'Medium', 'Low']
        course_opts = ['All'] + sorted(live['courseName'].dropna().unique().tolist())
        mod_opts    = ['All'] + sorted(live['lastModeratorName'].dropna().unique().tolist())

        sel_risk    = f1.selectbox("Risk level",              risk_opts,   key='lv_risk')
        sel_course  = f2.selectbox("Course",                  course_opts, key='lv_course')
        sel_mod     = f3.selectbox("My students (moderator)", mod_opts,    key='lv_mod')

        s1, s2 = st.columns(2)
        name_opts  = sorted(live['studentName'].dropna().unique().tolist())
        phone_opts = sorted(live['phoneNumber'].dropna().astype(str).unique().tolist())
        sel_name   = s1.selectbox("Search by student name",  name_opts,  key='lv_name',  index=None, placeholder="Type a name...")
        sel_phone  = s2.selectbox("Search by phone number",  phone_opts, key='lv_phone', index=None, placeholder="Type a phone number...")

        view = live.copy()
        if sel_risk   != 'All': view = view[view['risk_level']        == sel_risk]
        if sel_course != 'All': view = view[view['courseName']        == sel_course]
        if sel_mod    != 'All': view = view[view['lastModeratorName'] == sel_mod]
        if sel_name   is not None: view = view[view['studentName']              == sel_name]
        if sel_phone  is not None: view = view[view['phoneNumber'].astype(str)  == sel_phone]

        st.caption(f"Showing {len(view):,} of {len(live):,} active students")

        # ---- risk distribution by course --------------------
        if len(view) > 0:
            dist = view.groupby(['courseName', 'risk_level']).size().reset_index(name='n')
            fig = px.bar(
                dist, x='courseName', y='n', color='risk_level',
                color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'},
                title="Risk distribution by course",
                labels={'n': 'Students', 'courseName': 'Course'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- ranked table -----------------------------------
        def colour_risk(val):
            return {
                'High':   'background-color: #e74c3c; color: white',
                'Medium': 'background-color: #f39c12; color: white',
                'Low':    'background-color: #2ecc71; color: white',
            }.get(val, '')

        table = view.sort_values('dropout_probability', ascending=False)[[
            'studentName', 'phoneNumber', 'courseName',
            'dropout_probability', 'risk_level', 'topFactors',
            'lastTeacherName', 'lastModeratorName'
        ]].rename(columns={
            'studentName': 'Student', 'phoneNumber': 'Phone',
            'courseName': 'Course', 'dropout_probability': 'Risk %',
            'risk_level': 'Risk', 'topFactors': 'Top Factors',
            'lastTeacherName': 'Teacher', 'lastModeratorName': 'Moderator'
        }).reset_index(drop=True)
        table['Risk %'] = (table['Risk %'] * 100).round(1)

        st.dataframe(
            table.style.map(colour_risk, subset=['Risk'])
                       .format({'Risk %': '{:.1f}'}),
            use_container_width=True, height=480
        )

        st.download_button(
            "Download live predictions as CSV",
            data=table.to_csv(index=False).encode('utf-8'),
            file_name="live_predictions.csv",
            mime="text/csv"
        )