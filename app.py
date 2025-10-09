import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Strategic Workforce Planning Dashboard", layout="wide")

# Generate mock data
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
        'Exits': np.random.choice([0, 1], n_employees, p=[0.9, 0.1]),
        'Moves_Out': np.random.choice([0, 1], n_employees, p=[0.95, 0.05]),
        'Hires': 0,
        'Moves_In': 0,
        'Predicted_Exits': np.random.choice([0, 1], n_employees, p=[0.92, 0.08])
    })
    
    # Add some hires and moves
    n_hires = 50
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
        'Moves_In': np.random.choice([0, 1], n_hires, p=[0.7, 0.3]),
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
    
    # Define colors for each category
    colors = {
        'Established': '#FF6B6B',  # Light red
        'Peaking': '#4ECDC4',      # Teal
        'Developing': '#45B7D1',   # Blue
        'Rising': '#96CEB4'        # Green
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
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#d62728"}},
        increasing={"marker": {"color": "#2ca02c"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))
    
    # Add target line using layout shapes instead of add_hline
    fig_bridge.add_shape(
        type="line",
        x0=0, x1=1,
        y0=target, y1=target,
        xref="paper", yref="y",
        line=dict(dash="dash", color="orange", width=2)
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
        arrowcolor="orange",
        ax=-20, ay=-30,
        font=dict(size=12, color="orange")
    )
    
    fig_bridge.update_layout(
        height=600,
        showlegend=False,
        yaxis_title="Employee Count"
    )
    
    st.plotly_chart(fig_bridge, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# Section 2: Detailed Breakdown Table
st.markdown("#### 📋 Detailed Workforce Movement Breakdown")

# Aggregate data by hierarchy
def create_hierarchy_table(df):
    # SBU Level
    sbu_agg = df.groupby('SBU').agg({
        'Current_Employee': 'sum',
        'Exits': 'sum',
        'Moves_Out': 'sum',
        'Hires': 'sum',
        'Moves_In': 'sum',
        'Predicted_Exits': 'sum'
    }).reset_index()
    sbu_agg['Level'] = 'SBU'
    sbu_agg['Name'] = sbu_agg['SBU']
    sbu_agg['Parent'] = ''
    
    # Company Level
    company_agg = df.groupby(['SBU', 'Company']).agg({
        'Current_Employee': 'sum',
        'Exits': 'sum',
        'Moves_Out': 'sum',
        'Hires': 'sum',
        'Moves_In': 'sum',
        'Predicted_Exits': 'sum'
    }).reset_index()
    company_agg['Level'] = 'Company'
    company_agg['Name'] = company_agg['Company']
    company_agg['Parent'] = company_agg['SBU']
    
    # Cost Center Level
    cc_agg = df.groupby(['SBU', 'Company', 'Cost_Center']).agg({
        'Current_Employee': 'sum',
        'Exits': 'sum',
        'Moves_Out': 'sum',
        'Hires': 'sum',
        'Moves_In': 'sum',
        'Predicted_Exits': 'sum'
    }).reset_index()
    cc_agg['Level'] = 'Cost Center'
    cc_agg['Name'] = cc_agg['Cost_Center']
    cc_agg['Parent'] = cc_agg['Company']
    
    # Employee Level
    emp_agg = df[['Employee_ID', 'SBU', 'Company', 'Cost_Center', 'Current_Employee', 
                  'Exits', 'Moves_Out', 'Hires', 'Moves_In', 'Predicted_Exits']].copy()
    emp_agg['Level'] = 'Employee'
    emp_agg['Name'] = emp_agg['Employee_ID']
    emp_agg['Parent'] = emp_agg['Cost_Center']
    
    return sbu_agg, company_agg, cc_agg, emp_agg

sbu_agg, company_agg, cc_agg, emp_agg = create_hierarchy_table(filtered_data)

# Calculate projected and all targets for each level
def calculate_targets(df_level, base_count):
    df_level['Projected'] = (df_level['Current_Employee'] - df_level['Exits'] - 
                              df_level['Moves_Out'] + df_level['Hires'] + 
                              df_level['Moves_In'] - df_level['Predicted_Exits'])
    
    # Calculate all three target types
    df_level['Re_Org_Target'] = (df_level['Current_Employee'] * 0.95).astype(int)
    df_level['AI_Target'] = (df_level['Current_Employee'] * 0.90).astype(int)
    
    # Demand target only for commercial and operational jobs
    # Check if this level has commercial/operational jobs
    commercial_operational = ['Sales', 'Operations']  # Job family groups that are commercial/operational
    
    # For mock data, we'll apply demand targets to all levels
    # In real implementation, this would check actual job family group distribution per level
    # and only apply demand targets where commercial/operational roles exist
    df_level['Demand_Target'] = (df_level['Current_Employee'] * 1.05).astype(int)
    
    # Note: In production, you would filter by job family groups like this:
    # commercial_ops_mask = df_level['Job_Family_Group'].isin(commercial_operational)
    # df_level.loc[commercial_ops_mask, 'Demand_Target'] = (df_level.loc[commercial_ops_mask, 'Current_Employee'] * 1.05).astype(int)
    # df_level.loc[~commercial_ops_mask, 'Demand_Target'] = df_level.loc[~commercial_ops_mask, 'Current_Employee']  # No change for non-commercial
    
    # Calculate deltas for all targets
    df_level['Re_Org_Delta'] = df_level['Projected'] - df_level['Re_Org_Target']
    df_level['AI_Delta'] = df_level['Projected'] - df_level['AI_Target']
    df_level['Demand_Delta'] = df_level['Projected'] - df_level['Demand_Target']
    
    # Add indicators for each target
    def get_indicator(delta):
        if abs(delta) <= 2:  # Within 2 FTE is "about right"
            return "✅ About Right"
        elif delta > 0:
            return "⚠️ Above Target (Too Much)"
        else:
            return "🔴 Below Target (Too Little)"
    
    df_level['Re_Org_Indicator'] = df_level['Re_Org_Delta'].apply(get_indicator)
    df_level['AI_Indicator'] = df_level['AI_Delta'].apply(get_indicator)
    df_level['Demand_Indicator'] = df_level['Demand_Delta'].apply(get_indicator)
    
    return df_level

# Apply target calculations to all levels
sbu_agg = calculate_targets(sbu_agg, current_count)
company_agg = calculate_targets(company_agg, current_count)
cc_agg = calculate_targets(cc_agg, current_count)
emp_agg = calculate_targets(emp_agg, current_count)

# Add employee count summaries for higher aggregation levels
def add_employee_summaries(company_agg, cc_agg, sbu_agg):
    # Function to categorize employees based on delta
    def categorize_employees(delta):
        if delta > 2:  # More than 2 FTE above target
            return {'too_many': abs(delta), 'about_right': 0, 'too_little': 0}
        elif delta < -2:  # More than 2 FTE below target
            return {'too_many': 0, 'about_right': 0, 'too_little': abs(delta)}
        else:  # Within ±2 FTE
            return {'too_many': 0, 'about_right': abs(delta) if abs(delta) <= 2 else 2, 'too_little': 0}
    
    # Calculate employee counts per company (sum of all cost centers)
    cc_employee_counts = cc_agg.groupby(['SBU', 'Company']).agg({
        'Re_Org_Delta': lambda x: sum([categorize_employees(d)['too_many'] for d in x]),
        'AI_Delta': lambda x: sum([categorize_employees(d)['too_many'] for d in x]),
        'Demand_Delta': lambda x: sum([categorize_employees(d)['too_many'] for d in x])
    }).reset_index()
    
    # Add about right and too little counts
    for target in ['Re_Org', 'AI', 'Demand']:
        cc_employee_counts[f'{target}_About_Right'] = cc_agg.groupby(['SBU', 'Company'])[f'{target}_Delta'].apply(
            lambda x: sum([categorize_employees(d)['about_right'] for d in x])
        ).reset_index()[f'{target}_Delta']
        cc_employee_counts[f'{target}_Too_Little'] = cc_agg.groupby(['SBU', 'Company'])[f'{target}_Delta'].apply(
            lambda x: sum([categorize_employees(d)['too_little'] for d in x])
        ).reset_index()[f'{target}_Delta']
    
    # Rename columns
    cc_employee_counts.columns = ['SBU', 'Company', 'Re_Org_Too_Many', 'AI_Too_Many', 'Demand_Too_Many',
                                 'Re_Org_About_Right', 'AI_About_Right', 'Demand_About_Right',
                                 'Re_Org_Too_Little', 'AI_Too_Little', 'Demand_Too_Little']
    
    # Merge with company data
    company_agg = company_agg.merge(cc_employee_counts, on=['SBU', 'Company'], how='left')
    
    # Calculate company employee counts per SBU
    company_employee_counts = company_agg.groupby('SBU').agg({
        'Re_Org_Too_Many': 'sum', 'Re_Org_About_Right': 'sum', 'Re_Org_Too_Little': 'sum',
        'AI_Too_Many': 'sum', 'AI_About_Right': 'sum', 'AI_Too_Little': 'sum',
        'Demand_Too_Many': 'sum', 'Demand_About_Right': 'sum', 'Demand_Too_Little': 'sum'
    }).reset_index()
    
    # Rename for SBU level
    company_employee_counts.columns = ['SBU', 'SBU_Re_Org_Too_Many', 'SBU_Re_Org_About_Right', 'SBU_Re_Org_Too_Little',
                                      'SBU_AI_Too_Many', 'SBU_AI_About_Right', 'SBU_AI_Too_Little',
                                      'SBU_Demand_Too_Many', 'SBU_Demand_About_Right', 'SBU_Demand_Too_Little']
    
    # Merge with SBU data
    sbu_agg = sbu_agg.merge(company_employee_counts, on='SBU', how='left')
    
    return sbu_agg, company_agg

sbu_agg, company_agg = add_employee_summaries(company_agg, cc_agg, sbu_agg)

# Status Summary Section
st.markdown("#### 📊 Employee Count Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🔄 Re-Organization Target")
    # Calculate totals across all SBUs
    too_many = sbu_agg['SBU_Re_Org_Too_Many'].sum() if 'SBU_Re_Org_Too_Many' in sbu_agg.columns else 0
    about_right = sbu_agg['SBU_Re_Org_About_Right'].sum() if 'SBU_Re_Org_About_Right' in sbu_agg.columns else 0
    too_little = sbu_agg['SBU_Re_Org_Too_Little'].sum() if 'SBU_Re_Org_Too_Little' in sbu_agg.columns else 0
    
    st.metric("⚠️ Too Many", too_many)
    st.metric("✅ About Right", about_right)
    st.metric("🔴 Too Little", too_little)

with col2:
    st.markdown("##### 🤖 AI Target")
    # Calculate totals across all SBUs
    too_many = sbu_agg['SBU_AI_Too_Many'].sum() if 'SBU_AI_Too_Many' in sbu_agg.columns else 0
    about_right = sbu_agg['SBU_AI_About_Right'].sum() if 'SBU_AI_About_Right' in sbu_agg.columns else 0
    too_little = sbu_agg['SBU_AI_Too_Little'].sum() if 'SBU_AI_Too_Little' in sbu_agg.columns else 0
    
    st.metric("⚠️ Too Many", too_many)
    st.metric("✅ About Right", about_right)
    st.metric("🔴 Too Little", too_little)

with col3:
    st.markdown("##### 📈 Demand Target")
    # Calculate totals across all SBUs
    too_many = sbu_agg['SBU_Demand_Too_Many'].sum() if 'SBU_Demand_Too_Many' in sbu_agg.columns else 0
    about_right = sbu_agg['SBU_Demand_About_Right'].sum() if 'SBU_Demand_About_Right' in sbu_agg.columns else 0
    too_little = sbu_agg['SBU_Demand_Too_Little'].sum() if 'SBU_Demand_Too_Little' in sbu_agg.columns else 0
    
    st.metric("⚠️ Too Many", too_many)
    st.metric("✅ About Right", about_right)
    st.metric("🔴 Too Little", too_little)

st.markdown("---")

# Display hierarchical view with expanders
st.markdown("##### 🏢 SBU Level Summary")
st.markdown("**Note**: Shows actual employee counts that are too many, about right, or too little for each target type.")

sbu_display = sbu_agg[['Name', 'Current_Employee', 'Exits', 'Moves_Out', 'Hires', 
                         'Moves_In', 'Predicted_Exits', 'Projected', 
                         'Re_Org_Target', 'SBU_Re_Org_Too_Many', 'SBU_Re_Org_About_Right', 'SBU_Re_Org_Too_Little',
                         'AI_Target', 'SBU_AI_Too_Many', 'SBU_AI_About_Right', 'SBU_AI_Too_Little',
                         'Demand_Target', 'SBU_Demand_Too_Many', 'SBU_Demand_About_Right', 'SBU_Demand_Too_Little']].copy()
sbu_display.columns = ['SBU', 'Current', 'Exits', 'Moves Out', 'Hires', 
                        'Moves In', 'Predicted Exits', 'Projected',
                        'Re-Org Target', 'Re-Org Too Many', 'Re-Org About Right', 'Re-Org Too Little',
                        'AI Target', 'AI Too Many', 'AI About Right', 'AI Too Little',
                        'Demand Target', 'Demand Too Many', 'Demand About Right', 'Demand Too Little']
st.dataframe(sbu_display, use_container_width=True, hide_index=True)

# Expandable views for each SBU
for sbu in sbu_agg['SBU'].unique():
    with st.expander(f"📂 {sbu} - Detailed View"):
        # Company level for this SBU
        company_subset = company_agg[company_agg['SBU'] == sbu]
        st.markdown(f"**Companies in {sbu}**")
        company_display = company_subset[['Name', 'Current_Employee', 'Exits', 'Moves_Out', 'Hires', 
                                           'Moves_In', 'Predicted_Exits', 'Projected',
                                           'Re_Org_Target', 'Re_Org_Too_Many', 'Re_Org_About_Right', 'Re_Org_Too_Little',
                                           'AI_Target', 'AI_Too_Many', 'AI_About_Right', 'AI_Too_Little',
                                           'Demand_Target', 'Demand_Too_Many', 'Demand_About_Right', 'Demand_Too_Little']].copy()
        company_display.columns = ['Company', 'Current', 'Exits', 'Moves Out', 'Hires', 
                                    'Moves In', 'Predicted Exits', 'Projected',
                                    'Re-Org Target', 'Re-Org Too Many', 'Re-Org About Right', 'Re-Org Too Little',
                                    'AI Target', 'AI Too Many', 'AI About Right', 'AI Too Little',
                                    'Demand Target', 'Demand Too Many', 'Demand About Right', 'Demand Too Little']
        st.dataframe(company_display, use_container_width=True, hide_index=True)
        
        # Cost center level
        for company in company_subset['Company'].unique():
            with st.expander(f"  📁 {company} - Cost Centers"):
                cc_subset = cc_agg[(cc_agg['SBU'] == sbu) & (cc_agg['Company'] == company)]
                cc_display = cc_subset[['Name', 'Current_Employee', 'Exits', 'Moves_Out', 'Hires', 
                                         'Moves_In', 'Predicted_Exits', 'Projected',
                                         'Re_Org_Target', 'Re_Org_Delta', 'Re_Org_Indicator',
                                         'AI_Target', 'AI_Delta', 'AI_Indicator',
                                         'Demand_Target', 'Demand_Delta', 'Demand_Indicator']].copy()
                cc_display.columns = ['Cost Center', 'Current', 'Exits', 'Moves Out', 'Hires', 
                                       'Moves In', 'Predicted Exits', 'Projected',
                                       'Re-Org Target', 'Re-Org Δ', 'Re-Org Status',
                                       'AI Target', 'AI Δ', 'AI Status', 
                                       'Demand Target', 'Demand Δ', 'Demand Status']
                st.dataframe(cc_display, use_container_width=True, hide_index=True)
                
                # Employee level (optional - can be very detailed)
                for cc in cc_subset['Cost_Center'].unique():
                    with st.expander(f"    👥 {cc} - Employees"):
                        emp_subset = emp_agg[(emp_agg['SBU'] == sbu) & 
                                              (emp_agg['Company'] == company) & 
                                              (emp_agg['Cost_Center'] == cc)]
                        emp_display = emp_subset[['Name', 'Current_Employee', 'Exits', 'Moves_Out', 
                                                   'Hires', 'Moves_In', 'Predicted_Exits', 'Projected',
                                                   'Re_Org_Target', 'Re_Org_Delta', 'Re_Org_Indicator',
                                                   'AI_Target', 'AI_Delta', 'AI_Indicator',
                                                   'Demand_Target', 'Demand_Delta', 'Demand_Indicator']].copy()
                        emp_display.columns = ['Employee ID', 'Current', 'Exits', 'Moves Out', 'Hires', 
                                                'Moves In', 'Predicted Exits', 'Projected',
                                                'Re-Org Target', 'Re-Org Δ', 'Re-Org Status',
                                                'AI Target', 'AI Δ', 'AI Status', 
                                                'Demand Target', 'Demand Δ', 'Demand Status']
                        st.dataframe(emp_display, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("💡 This is a mock-up dashboard with generated data for demonstration purposes.")

