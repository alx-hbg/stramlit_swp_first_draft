import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder

# Page config
st.set_page_config(page_title="Strategic Workforce Planning Dashboard", layout="wide")

# Apply Zeppelin styling to the Streamlit interface
st.markdown("""
<style>
    /* Main app styling with Zeppelin colors */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* Headers with Zeppelin colors */
    h1 {
        color: #27166F !important;
        font-family: Arial, sans-serif !important;
    }
    
    h2, h3, h4, h5, h6 {
        color: #193A68 !important;
        font-family: Arial, sans-serif !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #8C8C8C;
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="metric-value"] {
        color: #27166F !important;
        font-weight: bold !important;
    }
    
    [data-testid="metric-label"] {
        color: #193A68 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons and controls */
    .stButton > button {
        background-color: #00A5DD;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
    }
    
    .stButton > button:hover {
        background-color: #193A68;
    }
    
    /* Selectbox and multiselect styling */
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border: 1px solid #8C8C8C;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #B0D6F2;
        color: #193A68;
        font-weight: bold;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #8C8C8C;
    }
</style>
""", unsafe_allow_html=True)

# Generate mock data with more logical workforce metrics
@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    n_employees = 500
    
    sbus = ['SBU North', 'SBU South', 'SBU East', 'SBU West', 'SBU Central']
    companies = ['Company A', 'Company B', 'Company C', 'Company D']
    cost_centers = [f'CC-{i:03d}' for i in range(1, 21)]
    managers = [f'Manager {i}' for i in range(1, 31)]
    job_family_groups = ['Engineering', 'Sales', 'Operations', 'Finance', 'HR', 'IT']
    job_families = ['Software Dev', 'Product', 'Account Mgmt', 'Supply Chain', 'Accounting', 
                    'Recruiting', 'Infrastructure', 'Data Science', 'Marketing', 'Legal']
    job_profiles = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Principal', 'Manager', 'Director']
    grades = [f'Grade {i}' for i in range(1, 26)]  # Grades 1-25
    
    employees = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'SBU': np.random.choice(sbus, n_employees),
        'Company': np.random.choice(companies, n_employees),
        'Cost_Center': np.random.choice(cost_centers, n_employees),
        'Manager': np.random.choice(managers, n_employees),
        'Job_Family_Group': np.random.choice(job_family_groups, n_employees),
        'Job_Family': np.random.choice(job_families, n_employees),
        'Job_Profile': np.random.choice(job_profiles, n_employees),
        'Grade': np.random.choice(grades, n_employees),
        'Grade_Progression': np.random.uniform(-2, 2, n_employees),
        'Current_Employee': 1,
        'Exits': np.random.choice([0, 1], n_employees, p=[0.85, 0.15]),  # 15% exit rate
        'Moves_Out': np.random.choice([0, 1], n_employees, p=[0.92, 0.08]),  # 8% move out rate
        'Hires': 0,
        'Moves_In': 0,
        'Predicted_Exits': np.random.choice([0, 1], n_employees, p=[0.88, 0.12])  # 12% predicted exit rate
    })
    
    # Add some hires and moves with more realistic numbers
    n_hires = 75  # 15% hire rate to offset exits
    hires_df = pd.DataFrame({
        'Employee_ID': [f'NEW{i:04d}' for i in range(1, n_hires + 1)],
        'SBU': np.random.choice(sbus, n_hires),
        'Company': np.random.choice(companies, n_hires),
        'Cost_Center': np.random.choice(cost_centers, n_hires),
        'Manager': np.random.choice(managers, n_hires),
        'Job_Family_Group': np.random.choice(job_family_groups, n_hires),
        'Job_Family': np.random.choice(job_families, n_hires),
        'Job_Profile': np.random.choice(job_profiles, n_hires),
        'Grade': np.random.choice(grades, n_hires),
        'Grade_Progression': np.random.uniform(-2, 2, n_hires),
        'Current_Employee': 0,
        'Exits': 0,
        'Moves_Out': 0,
        'Hires': 1,
        'Moves_In': np.random.choice([0, 1], n_hires, p=[0.6, 0.4]),  # 40% are moves in
        'Predicted_Exits': 0
    })
    
    all_data = pd.concat([employees, hires_df], ignore_index=True)
    return all_data

data = generate_mock_data()

# Title
st.title("📊 Strategic Workforce Planning Dashboard")
st.markdown("**Workday-committed Data and Sales/Service Demand drive a 12-Month Forecast while Machine Learning on Employee Histories projects 5-Year Outcomes against Re-Org, AI, and Demand Targets**")

# Add explainer section
with st.expander("📋 Strategic Workforce Planning Framework - Click to understand the methodology", expanded=False):
    st.markdown("""
    ### 🎯 **Strategic Workforce Planning Overview**
    
    This dashboard implements a comprehensive strategic workforce planning framework that combines **Workday data**, **sales/service demand**, and **machine learning predictions** to provide actionable insights for workforce management.
    
    #### **📊 PROGRESSION ANALYTICS**
    The Grade Progression Matrix provides insights into employee development and career trajectories:
    
    - **🟢 Rising** (Grades 1-12 + Positive Progression): Emerging talent showing growth potential
    - **🔵 Developing** (Grades 1-12 + Negative Progression): Talent requiring development support
    - **🔴 Established** (Grades 13-25 + Negative Progression): Senior talent that may need engagement
    - **🟡 Peaking** (Grades 13-25 + Positive Progression): Top performers at their career peak
    
    **Manager Actions**: Coaching, Development Plans, Mentoring, Burnout Prevention, Retention, Succession Planning
    
    #### **🔮 FORECAST AND PREDICTION**
    
    **Committed to Join/Leave Data:**
    - Built entirely from Workday records (approved/scheduled)
    - Planned start dates from signed offers
    - Retirement dates and fixed-term end dates
    - Planned return dates from leave records
    
    **ML-based Exit and Mobility Predictions:**
    - Machine learning assigns voluntary exit probability and risk windows (12/24/36/48 months)
    - Internal mobility probabilities (promotion/lateral moves)
    - Aggregated expected exits/moves by month/team
    - High-risk roles/locations highlighted
    
    #### **🎯 TARGETS**
    
    **1. Committed Target Operating Model:**
    - Approved target activities and FTE as baseline
    - Activities connected with Job Profile and Cost Center
    - Phased ramp to target structure (years)
    - Over/understaffing surfaced by Org Unit and Job Profile
    
    **2. Anticipated AI Impact:**
    - Per-Job-Profile AI/automation potential from external sources (O*NET, ESCO)
    - Potential converted into FTE capacity (redeploy vs. reduce)
    - Missed productivity gains and residual over/understaffing highlighted
    
    **3. Expected Sales Revenue and Service Demand:**
    - Historic ratios built on € revenue per FTE and service workload per FTE
    - Future revenue and demand projected from pipeline (Salesforce, etc.)
    - Demand translated into FTE via past utilization & staffing ratios
    - **Note**: Demand targets are only relevant for Commercial and Operational job roles
    """)

st.markdown("---")

# Filters Section
st.sidebar.header("🔍 Filters")

# Time View Filter
time_view = st.sidebar.selectbox(
    "Time View",
    ["Current Year (until 31.12)", "12 Months Ahead", "5 Years Ahead"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Organizational Filters")

# SBU Filter
sbu_options = ['All'] + sorted(data['SBU'].unique().tolist())
selected_sbus = st.sidebar.multiselect(
    "SBU",
    options=sbu_options,
    default=['All']
)

# Company Filter
company_options = ['All'] + sorted(data['Company'].unique().tolist())
selected_companies = st.sidebar.multiselect(
    "Company",
    options=company_options,
    default=['All']
)

# Cost Center Filter
cost_center_options = ['All'] + sorted(data['Cost_Center'].unique().tolist())
selected_cost_centers = st.sidebar.multiselect(
    "Cost Center",
    options=cost_center_options,
    default=['All']
)

# Manager Filter
manager_options = ['All'] + sorted(data['Manager'].unique().tolist())
selected_managers = st.sidebar.multiselect(
    "Manager",
    options=manager_options,
    default=['All']
)

st.sidebar.markdown("---")
st.sidebar.subheader("Job Structure Filters")

# Job Family Group Filter
jfg_options = ['All'] + sorted(data['Job_Family_Group'].unique().tolist())
selected_jfgs = st.sidebar.multiselect(
    "Job Family Group",
    options=jfg_options,
    default=['All']
)

# Job Family Filter
jf_options = ['All'] + sorted(data['Job_Family'].unique().tolist())
selected_jfs = st.sidebar.multiselect(
    "Job Family",
    options=jf_options,
    default=['All']
)

# Job Profile Filter
jp_options = ['All'] + sorted(data['Job_Profile'].unique().tolist())
selected_jps = st.sidebar.multiselect(
    "Job Profile",
    options=jp_options,
    default=['All']
)

# Apply filters
filtered_data = data.copy()

if 'All' not in selected_sbus:
    filtered_data = filtered_data[filtered_data['SBU'].isin(selected_sbus)]
if 'All' not in selected_companies:
    filtered_data = filtered_data[filtered_data['Company'].isin(selected_companies)]
if 'All' not in selected_cost_centers:
    filtered_data = filtered_data[filtered_data['Cost_Center'].isin(selected_cost_centers)]
if 'All' not in selected_managers:
    filtered_data = filtered_data[filtered_data['Manager'].isin(selected_managers)]
if 'All' not in selected_jfgs:
    filtered_data = filtered_data[filtered_data['Job_Family_Group'].isin(selected_jfgs)]
if 'All' not in selected_jfs:
    filtered_data = filtered_data[filtered_data['Job_Family'].isin(selected_jfs)]
if 'All' not in selected_jps:
    filtered_data = filtered_data[filtered_data['Job_Profile'].isin(selected_jps)]

# Target Type Selection
st.sidebar.markdown("---")
target_type = st.sidebar.selectbox(
    "Target Type",
    ["Re-Org Target", "Demand Target", "AI Target"]
)

# Main Dashboard
st.subheader(f"📅 View: {time_view} | 🎯 {target_type}")
st.markdown("---")

# Section 1: Grade Matrix and Bridge Diagram
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Grade Progression Matrix")
    
    # Create grade progression matrix with individual employee dots
    def categorize_employee(row):
        grade_num = int(row['Grade'].split()[-1])
        progression = row['Grade_Progression']
        
        # High/Low grade: grades 13-25 are high, 1-12 are low
        # Positive/Negative progression
        if grade_num >= 13:  # High grade (grades 13-25)
            if progression >= 0:
                return 'Peaking'
            else:
                return 'Established'
        else:  # Low grade (grades 1-12)
            if progression >= 0:
                return 'Rising'
            else:
                return 'Developing'
    
    filtered_data['Category'] = filtered_data.apply(categorize_employee, axis=1)
    
    # Create scatter plot with individual employee dots
    fig_matrix = go.Figure()
    
    # Define colors for each category using Zeppelin color scheme
    colors = {
        'Established': '#AF0E0E',  # Zeppelin Red
        'Peaking': '#00A5DD',      # Zeppelin Cyan
        'Developing': '#193A68',   # Zeppelin Dark Blue
        'Rising': '#82368C'        # Zeppelin Purple
    }
    
    # Add scatter points for each category
    for category in ['Established', 'Peaking', 'Developing', 'Rising']:
        category_data = filtered_data[filtered_data['Category'] == category]
        if len(category_data) > 0:
            # Convert grade to numeric for plotting
            grade_nums = category_data['Grade'].str.split().str[-1].astype(int)
            progressions = category_data['Grade_Progression']
            
            fig_matrix.add_trace(go.Scatter(
                x=progressions,
                y=grade_nums,
                mode='markers',
                marker=dict(
                    color=colors[category],
                    size=8,
                    opacity=0.7
                ),
                name=category,
                text=category_data['Employee_ID'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Grade: %{y}<br>' +
                             'Progression: %{x:.2f}<br>' +
                             'Category: ' + category +
                             '<extra></extra>'
            ))
    
    # Add quadrant background colors (updated for 25 grades)
    fig_matrix.add_shape(
        type="rect",
        x0=-2, y0=12.5, x1=0, y1=25.5,
        fillcolor=colors['Established'],
        opacity=0.1,
        line=dict(width=0)
    )
    fig_matrix.add_shape(
        type="rect",
        x0=0, y0=12.5, x1=2, y1=25.5,
        fillcolor=colors['Peaking'],
        opacity=0.1,
        line=dict(width=0)
    )
    fig_matrix.add_shape(
        type="rect",
        x0=-2, y0=0.5, x1=0, y1=12.5,
        fillcolor=colors['Developing'],
        opacity=0.1,
        line=dict(width=0)
    )
    fig_matrix.add_shape(
        type="rect",
        x0=0, y0=0.5, x1=2, y1=12.5,
        fillcolor=colors['Rising'],
        opacity=0.1,
        line=dict(width=0)
    )
    
    # Add quadrant labels using layout annotations instead of add_annotation
    quadrant_annotations = [
        dict(x=-1, y=19, text="Established", showarrow=False, font=dict(size=14, color="black")),
        dict(x=1, y=19, text="Peaking", showarrow=False, font=dict(size=14, color="black")),
        dict(x=-1, y=6.5, text="Developing", showarrow=False, font=dict(size=14, color="black")),
        dict(x=1, y=6.5, text="Rising", showarrow=False, font=dict(size=14, color="black"))
    ]
    
    fig_matrix.update_layout(
        height=600,
        xaxis_title="Grade Progression →",
        yaxis_title="Grade →",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        annotations=quadrant_annotations
    )
    
    st.plotly_chart(fig_matrix, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("#### 🌉 Workforce Bridge Diagram")
    
    # Calculate bridge components
    current_count = filtered_data['Current_Employee'].sum()
    exits = filtered_data['Exits'].sum()
    moves_out = filtered_data['Moves_Out'].sum()
    hires = filtered_data['Hires'].sum()
    moves_in = filtered_data['Moves_In'].sum()
    predicted_exits = filtered_data['Predicted_Exits'].sum()
    
    # Calculate projected count
    projected = current_count - exits - moves_out + hires + moves_in - predicted_exits
    
    # Set target based on target type (mock targets)
    targets = {
        "Re-Org Target": int(current_count * 0.95),
        "Demand Target": int(current_count * 1.05),
        "AI Target": int(current_count * 0.90)
    }
    target = targets[target_type]
    delta = projected - target
    
    # Create waterfall chart
    measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "total", "relative"]
    values = [current_count, -exits, -moves_out, hires, moves_in, -predicted_exits, projected, delta]
    labels = ["Current<br>Employees", "Exits", "Moves<br>Out", "Hires", "Moves<br>In", 
              "Predicted<br>Exits", "Projected", "Delta to<br>Target"]
    
    fig_bridge = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"{v:+.0f}" if v < 0 else f"{v:.0f}" for v in values],
        textposition="outside",
        connector={"line": {"color": "#8C8C8C"}},
        decreasing={"marker": {"color": "#AF0E0E"}},
        increasing={"marker": {"color": "#00A5DD"}},
        totals={"marker": {"color": "#27166F"}}
    ))
    
    # Add target line using layout shapes instead of add_hline
    fig_bridge.add_shape(
        type="line",
        x0=0, x1=1,
        y0=target, y1=target,
        xref="paper", yref="y",
        line=dict(dash="dash", color="#FFCC00", width=2)
    )
    
    # Add target annotation
    fig_bridge.add_annotation(
        x=0.95, y=target,
        xref="paper", yref="y",
        text=f"{target_type}: {target}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FFCC00",
        ax=-20, ay=-30,
        font=dict(size=12, color="#FFCC00")
    )
    
    fig_bridge.update_layout(
        height=600,
        showlegend=False,
        yaxis_title="Employee Count"
    )
    
    st.plotly_chart(fig_bridge, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# Section 2: Workforce Summary Metrics (CHRO-focused)
st.markdown("#### 📊 Workforce Summary Metrics")

# Calculate key workforce metrics
current_employees = filtered_data['Current_Employee'].sum()
exits = filtered_data['Exits'].sum()
moves_out = filtered_data['Moves_Out'].sum()
hires = filtered_data['Hires'].sum()
moves_in = filtered_data['Moves_In'].sum()
predicted_exits = filtered_data['Predicted_Exits'].sum()

# Calculate projected count
projected_count = current_employees - exits - moves_out + hires + moves_in - predicted_exits

# Calculate targets with more realistic CHRO logic
re_org_target = int(current_employees * 0.95)  # 5% reduction for reorganization
ai_target = int(current_employees * 0.85)     # 15% reduction due to AI automation
demand_target = int(current_employees * 1.08)  # 8% growth due to business demand

# Calculate deltas
re_org_delta = projected_count - re_org_target
ai_delta = projected_count - ai_target
demand_delta = projected_count - demand_target

# Display metrics in CHRO-friendly format
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Employees", 
        f"{current_employees:,}",
        help="Total current employee count"
    )

with col2:
    st.metric(
        "Projected Count", 
        f"{projected_count:,}",
        delta=projected_count - current_employees,
        help="Expected headcount after planned movements"
    )

with col3:
    st.metric(
        "Re-Org Target", 
        f"{re_org_target:,}",
        delta=re_org_delta,
        help="Target headcount after reorganization (5% reduction)"
    )

with col4:
    st.metric(
        "AI Target", 
        f"{ai_target:,}",
        delta=ai_delta,
        help="Target headcount considering AI automation impact (15% reduction)"
    )

# Additional CHRO insights
st.markdown("**Key Workforce Insights:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Turnover Rate:** {(exits/current_employees*100):.1f}% (Target: <12%)")

with col2:
    st.info(f"**Hiring Rate:** {(hires/current_employees*100):.1f}% (Target: 15-20%)")

with col3:
    st.info(f"**Predicted Risk:** {(predicted_exits/current_employees*100):.1f}% at risk of leaving")

st.markdown("---")

# Section 3: AG Grid PivotTable Implementation - Following the Blog Post Exactly
st.markdown("#### 📋 Workforce Data PivotTable")

# Prepare data for AG Grid - include all previous table logic
pivot_data = filtered_data.copy()

# Add calculated fields from previous version
pivot_data['Projected_Count'] = (pivot_data['Current_Employee'] - pivot_data['Exits'] - 
                                 pivot_data['Moves_Out'] + pivot_data['Hires'] + 
                                 pivot_data['Moves_In'] - pivot_data['Predicted_Exits'])

# Add target calculations with realistic CHRO logic
pivot_data['Re_Org_Target'] = (pivot_data['Current_Employee'] * 0.95).astype(int)  # 5% reduction
pivot_data['AI_Target'] = (pivot_data['Current_Employee'] * 0.85).astype(int)     # 15% reduction
pivot_data['Demand_Target'] = (pivot_data['Current_Employee'] * 1.08).astype(int)  # 8% growth

# Add delta calculations
pivot_data['Re_Org_Delta'] = pivot_data['Projected_Count'] - pivot_data['Re_Org_Target']
pivot_data['AI_Delta'] = pivot_data['Projected_Count'] - pivot_data['AI_Target']
pivot_data['Demand_Delta'] = pivot_data['Projected_Count'] - pivot_data['Demand_Target']

# Add risk indicators from previous version
def get_risk_level(row):
    max_delta = max(abs(row['Re_Org_Delta']), abs(row['AI_Delta']), abs(row['Demand_Delta']))
    if max_delta <= 2:
        return "Low Risk"
    elif max_delta <= 5:
        return "Medium Risk"
    else:
        return "High Risk"

pivot_data['Risk_Level'] = pivot_data.apply(get_risk_level, axis=1)

# Add movement calculations
pivot_data['Total_Movement'] = (pivot_data['Exits'] + pivot_data['Moves_Out'] + 
                                pivot_data['Hires'] + pivot_data['Moves_In'])
pivot_data['Net_Movement'] = (pivot_data['Hires'] + pivot_data['Moves_In'] - 
                              pivot_data['Exits'] - pivot_data['Moves_Out'])

# Add Year and Month columns for pivoting
pivot_data['Year'] = 2024
pivot_data['Month'] = np.random.randint(1, 13, len(pivot_data))
pivot_data['Quarter'] = ((pivot_data['Month'] - 1) // 3) + 1

# Add employee category for CHRO insights
def categorize_employee(row):
    grade_num = int(row['Grade'].split()[-1])
    progression = row['Grade_Progression']
    
    if grade_num >= 13:  # High grade (grades 13-25)
        if progression >= 0:
            return 'Peaking'
        else:
            return 'Established'
    else:  # Low grade (grades 1-12)
        if progression >= 0:
            return 'Rising'
        else:
            return 'Developing'

pivot_data['Category'] = pivot_data.apply(categorize_employee, axis=1)

# Pivot mode toggle - exactly like the blog post
shouldDisplayPivoted = st.checkbox("Enable Pivot Mode", help="Toggle to enable row grouping and column pivoting")

# Configure AG Grid following the blog post exactly
gb = GridOptionsBuilder.from_dataframe(pivot_data)

# Configure default columns - exactly like the blog post
gb.configure_default_column(
    resizable=True,
    filterable=True,
    sortable=True,
    editable=False,
    enablePivot=True,
    enableValue=True,
    enableRowGroup=True,
)

# Configure columns - following the blog post pattern with row grouping
gb.configure_column(field="SBU", header_name="SBU", width=120, rowGroup=shouldDisplayPivoted)
gb.configure_column(field="Company", header_name="Company", width=150, rowGroup=shouldDisplayPivoted)
gb.configure_column(field="Cost_Center", header_name="Cost Center", width=120, rowGroup=shouldDisplayPivoted)
gb.configure_column(field="Job_Family_Group", header_name="Job Family Group", width=150, rowGroup=shouldDisplayPivoted)
gb.configure_column(field="Job_Family", header_name="Job Family", width=150, rowGroup=shouldDisplayPivoted)
gb.configure_column(field="Job_Profile", header_name="Job Profile", width=120)
gb.configure_column(field="Grade", header_name="Grade", width=80)
gb.configure_column(field="Category", header_name="Talent Category", width=120)
gb.configure_column(field="Risk_Level", header_name="Risk Level", width=100)

# Configure pivot columns - exactly like the blog post
gb.configure_column(
    field="Month",
    header_name="Month",
    width=80,
    pivot=True,
    hide=True,
    valueGetter="data.Month"
)

gb.configure_column(
    field="Quarter",
    header_name="Quarter",
    width=80,
    pivot=True,
    hide=True,
    valueGetter="data.Quarter"
)

# Configure value columns with aggregation - exactly like the blog post
gb.configure_column(
    field="Current_Employee",
    header_name="Current Employees",
    width=120,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Projected_Count",
    header_name="Projected Count",
    width=120,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Re_Org_Delta",
    header_name="Re-Org Delta",
    width=100,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="AI_Delta",
    header_name="AI Delta",
    width=100,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Demand_Delta",
    header_name="Demand Delta",
    width=100,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Total_Movement",
    header_name="Total Movement",
    width=120,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Net_Movement",
    header_name="Net Movement",
    width=120,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Exits",
    header_name="Exits",
    width=80,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Hires",
    header_name="Hires",
    width=80,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

gb.configure_column(
    field="Predicted_Exits",
    header_name="At Risk",
    width=80,
    type=["numericColumn"],
    aggFunc="sum",
    valueFormatter="value.toLocaleString()",
)

# Enable AG Grid sidebar for pivot controls
gb.configure_side_bar()

# Configure grid options - exactly like the blog post
gb.configure_grid_options(
    tooltipShowDelay=0,
    pivotMode=shouldDisplayPivoted,
    autoGroupColumnDef=dict(
        minWidth=300,
        pinned="left",
        cellRendererParams=dict(suppressCount=True)
    )
)

# Build grid options
go = gb.build()

# Display the AG Grid - exactly like the blog post
AgGrid(
    pivot_data,
    gridOptions=go,
    height=600,
    enable_enterprise_modules=True,
)

# CHRO Action Items
st.markdown("---")
st.markdown("#### 🎯 CHRO Action Items")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**High Priority Actions:**")
    if ai_delta > 0:
        st.warning(f"⚠️ **AI Impact:** {ai_delta} employees above AI target - consider reskilling/transition planning")
    if re_org_delta > 10:
        st.warning(f"⚠️ **Re-Org:** {re_org_delta} employees above reorganization target - plan workforce reduction")
    if predicted_exits > current_employees * 0.15:
        st.error(f"🚨 **Retention Risk:** {(predicted_exits/current_employees*100):.1f}% at risk - implement retention strategies")

with col2:
    st.markdown("**Strategic Opportunities:**")
    if demand_delta > 0:
        st.success(f"📈 **Growth Ready:** {demand_delta} employees above demand target - scale operations")
    if projected_count > current_employees:
        st.info(f"📊 **Net Growth:** +{projected_count - current_employees} employees - ensure onboarding capacity")
    if hires > exits:
        st.info(f"✅ **Positive Flow:** Net +{hires - exits} employees - maintain hiring momentum")

st.markdown("---")
st.caption("💡 **CHRO Dashboard:** Strategic workforce planning with AI-driven insights for executive decision-making")

