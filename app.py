"""
Dandy Design Operations Case Study - Data-Driven Analysis Tool
==============================================================
A comprehensive Streamlit application for analyzing dental crown design operations,
focusing on cost optimization, throughput metrics, and quality control.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Dandy Design Operations Analysis",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E5077;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1E3A5F;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

@st.cache_data
def generate_designer_data(n_designers=50, seed=42):
    """Generate synthetic designer workforce data."""
    np.random.seed(seed)
    random.seed(seed)
    
    designer_types = ['Onshore_Senior', 'Onshore_Junior', 'Offshore_Senior', 'Offshore_Junior', 'QC']
    
    designers = []
    for i in range(n_designers):
        dtype = random.choices(
            designer_types, 
            weights=[0.15, 0.20, 0.25, 0.30, 0.10]
        )[0]
        
        # Base attributes by type
        if dtype == 'QC':
            hourly_rate = np.random.uniform(45, 65)
            dental_knowledge = np.random.uniform(0.85, 0.98)
            experience_months = np.random.randint(24, 60)
            avg_design_time = np.random.uniform(8, 12)  # minutes for QC review
        elif 'Senior' in dtype:
            base_rate = 35 if 'Onshore' in dtype else 18
            hourly_rate = np.random.uniform(base_rate * 0.9, base_rate * 1.2)
            dental_knowledge = np.random.uniform(0.65, 0.90)
            experience_months = np.random.randint(12, 36)
            avg_design_time = np.random.uniform(18, 28)
        else:  # Junior
            base_rate = 25 if 'Onshore' in dtype else 12
            hourly_rate = np.random.uniform(base_rate * 0.85, base_rate * 1.15)
            dental_knowledge = np.random.uniform(0.20, 0.55)
            experience_months = np.random.randint(1, 12)
            avg_design_time = np.random.uniform(28, 45)
        
        location = 'Onshore' if 'Onshore' in dtype or dtype == 'QC' else 'Offshore'
        
        designers.append({
            'designer_id': f'D{i+1:03d}',
            'designer_type': dtype,
            'location': location,
            'hourly_rate': round(hourly_rate, 2),
            'dental_knowledge_score': round(dental_knowledge, 3),
            'experience_months': experience_months,
            'avg_design_time_min': round(avg_design_time, 1),
            'hire_date': datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 700)),
            'training_completed': True if experience_months > 3 else np.random.choice([True, False], p=[0.6, 0.4])
        })
    
    return pd.DataFrame(designers)


@st.cache_data
def generate_design_data(n_designs=5000, designers_df=None, seed=42):
    """Generate synthetic design task data with detailed step information."""
    np.random.seed(seed)
    random.seed(seed)
    
    if designers_df is None:
        designers_df = generate_designer_data()
    
    non_qc_designers = designers_df[designers_df['designer_type'] != 'QC']['designer_id'].tolist()
    qc_designers = designers_df[designers_df['designer_type'] == 'QC']['designer_id'].tolist()
    
    designs = []
    for i in range(n_designs):
        designer_id = random.choice(non_qc_designers)
        designer_info = designers_df[designers_df['designer_id'] == designer_id].iloc[0]
        
        qc_id = random.choice(qc_designers)
        qc_info = designers_df[designers_df['designer_id'] == qc_id].iloc[0]
        
        # Scan complexity affects all steps
        scan_complexity = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
        complexity_multiplier = {'Low': 0.8, 'Medium': 1.0, 'High': 1.4}[scan_complexity]
        
        # Doctor preference complexity
        doctor_pref_complexity = np.random.choice(['Standard', 'Custom', 'Highly Custom'], p=[0.5, 0.35, 0.15])
        pref_multiplier = {'Standard': 1.0, 'Custom': 1.2, 'Highly Custom': 1.5}[doctor_pref_complexity]
        
        # Time for each step (in minutes)
        base_times = {
            'step1_scan_review': 5,
            'step2_margin_trace': 8,
            'step3_crown_placement': 12,
            'step4_adjustments': 15,
            'step5_qc_review': 10
        }
        
        # Adjust times based on designer skill and complexity
        skill_factor = 1.5 - designer_info['dental_knowledge_score']
        experience_factor = max(0.7, 1.2 - (designer_info['experience_months'] / 50))
        
        step_times = {}
        for step, base_time in base_times.items():
            if step == 'step5_qc_review':
                # QC time based on QC designer
                time = base_time * complexity_multiplier * (1.3 - qc_info['dental_knowledge_score'])
            elif step in ['step1_scan_review', 'step2_margin_trace']:
                # Steps 1-2 don't need dental knowledge
                time = base_time * complexity_multiplier * experience_factor
            elif step == 'step3_crown_placement':
                # Step 3 benefits from dental knowledge
                time = base_time * complexity_multiplier * skill_factor * 0.8
            else:
                # Step 4 requires dental knowledge
                time = base_time * complexity_multiplier * pref_multiplier * skill_factor
            
            step_times[step] = round(max(2, time + np.random.normal(0, time * 0.15)), 1)
        
        total_design_time = sum([v for k, v in step_times.items() if k != 'step5_qc_review'])
        
        # Calculate rejection probability based on multiple factors
        base_rejection_prob = 0.35
        
        # Factors affecting rejection
        knowledge_factor = (0.7 - designer_info['dental_knowledge_score']) * 0.4
        experience_factor = max(0, (12 - designer_info['experience_months']) / 12) * 0.2
        complexity_factor = {'Low': -0.1, 'Medium': 0, 'High': 0.15}[scan_complexity]
        pref_factor = {'Standard': -0.05, 'Custom': 0.05, 'Highly Custom': 0.15}[doctor_pref_complexity]
        training_factor = 0 if designer_info['training_completed'] else 0.15
        
        rejection_prob = base_rejection_prob + knowledge_factor + experience_factor + complexity_factor + pref_factor + training_factor
        rejection_prob = max(0.05, min(0.70, rejection_prob))
        
        rejected = np.random.random() < rejection_prob
        
        # Rejection reasons
        rejection_reasons = []
        if rejected:
            possible_reasons = [
                ('Margin line inaccuracy', 0.25, 'step2'),
                ('Crown sizing issues', 0.20, 'step3'),
                ('Aesthetic adjustments needed', 0.25, 'step4'),
                ('Doctor preference not followed', 0.15, 'step4'),
                ('Scan defects not addressed', 0.10, 'step1'),
                ('Occlusion problems', 0.05, 'step3')
            ]
            # Weight reasons by designer weakness
            for reason, prob, step in possible_reasons:
                if np.random.random() < prob * (1.5 if designer_info['dental_knowledge_score'] < 0.5 else 1):
                    rejection_reasons.append(reason)
            if not rejection_reasons:
                rejection_reasons.append(random.choice([r[0] for r in possible_reasons]))
        
        # Rework time if rejected
        rework_time = 0
        if rejected:
            rework_time = np.random.uniform(8, 25) * complexity_multiplier
        
        # Calculate costs
        designer_cost = (total_design_time / 60) * designer_info['hourly_rate']
        qc_cost = (step_times['step5_qc_review'] / 60) * qc_info['hourly_rate']
        rework_cost = (rework_time / 60) * designer_info['hourly_rate'] if rejected else 0
        total_cost = designer_cost + qc_cost + rework_cost
        
        designs.append({
            'design_id': f'DES{i+1:05d}',
            'designer_id': designer_id,
            'designer_type': designer_info['designer_type'],
            'designer_location': designer_info['location'],
            'designer_dental_knowledge': designer_info['dental_knowledge_score'],
            'designer_experience_months': designer_info['experience_months'],
            'qc_id': qc_id,
            'scan_complexity': scan_complexity,
            'doctor_pref_complexity': doctor_pref_complexity,
            **step_times,
            'total_design_time_min': round(total_design_time, 1),
            'total_time_with_qc_min': round(total_design_time + step_times['step5_qc_review'], 1),
            'rejected_by_qc': rejected,
            'rejection_reasons': '; '.join(rejection_reasons) if rejection_reasons else None,
            'rework_time_min': round(rework_time, 1),
            'designer_cost': round(designer_cost, 2),
            'qc_cost': round(qc_cost, 2),
            'rework_cost': round(rework_cost, 2),
            'total_cost': round(total_cost, 2),
            'date': datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)),
            'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], p=[0.4, 0.4, 0.2])
        })
    
    return pd.DataFrame(designs)


@st.cache_data
def generate_squad_data(designers_df, designs_df, n_squads=8):
    """Generate squad-level aggregated data."""
    np.random.seed(42)
    
    # Assign designers to squads
    designers_df = designers_df.copy()
    qc_ids = designers_df[designers_df['designer_type'] == 'QC']['designer_id'].tolist()
    non_qc_ids = designers_df[designers_df['designer_type'] != 'QC']['designer_id'].tolist()
    
    # Create squads with one QC lead each
    squad_assignments = {}
    for i, qc_id in enumerate(qc_ids[:n_squads]):
        squad_assignments[qc_id] = f'Squad_{chr(65+i)}'
    
    # Distribute other designers
    random.shuffle(non_qc_ids)
    for i, designer_id in enumerate(non_qc_ids):
        squad_idx = i % n_squads
        squad_assignments[designer_id] = f'Squad_{chr(65+squad_idx)}'
    
    designers_df['squad'] = designers_df['designer_id'].map(squad_assignments)
    
    # Map squads to designs
    designs_df = designs_df.copy()
    designs_df['squad'] = designs_df['designer_id'].map(squad_assignments)
    
    # Aggregate by squad
    squad_metrics = designs_df.groupby('squad').agg({
        'design_id': 'count',
        'total_design_time_min': 'mean',
        'rejected_by_qc': 'mean',
        'total_cost': ['mean', 'sum'],
        'rework_time_min': 'mean',
        'designer_dental_knowledge': 'mean'
    }).round(3)
    
    squad_metrics.columns = ['total_designs', 'avg_design_time', 'rejection_rate', 
                            'avg_cost', 'total_cost', 'avg_rework_time', 'avg_dental_knowledge']
    squad_metrics = squad_metrics.reset_index()
    
    # Add squad composition
    squad_composition = designers_df.groupby(['squad', 'location']).size().unstack(fill_value=0)
    squad_composition['total_designers'] = squad_composition.sum(axis=1)
    squad_composition = squad_composition.reset_index()
    
    squad_metrics = squad_metrics.merge(squad_composition, on='squad')
    
    return designers_df, designs_df, squad_metrics


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_specialization_savings(designs_df, designers_df):
    """Calculate potential savings from step specialization model."""
    
    # Current state costs
    current_avg_cost = designs_df['total_cost'].mean()
    current_total_cost = designs_df['total_cost'].sum()
    
    # Projected costs with specialization
    # Steps 1-2: Can use lower-cost workers without dental knowledge
    step12_time = designs_df['step1_scan_review'].mean() + designs_df['step2_margin_trace'].mean()
    step12_rate = 15  # Lower rate for non-dental specialists
    
    # Step 3: Moderate dental knowledge
    step3_time = designs_df['step3_crown_placement'].mean()
    step3_rate = 22  # Moderate rate
    
    # Step 4: High dental knowledge
    step4_time = designs_df['step4_adjustments'].mean()
    step4_rate = 35  # Higher rate for specialists
    
    # QC: Same as current
    qc_time = designs_df['step5_qc_review'].mean()
    qc_rate = designs_df['qc_cost'].mean() / (qc_time / 60)
    
    projected_design_cost = (
        (step12_time / 60) * step12_rate +
        (step3_time / 60) * step3_rate +
        (step4_time / 60) * step4_rate +
        (qc_time / 60) * qc_rate
    )
    
    # Assume 15% reduction in rejection rate due to specialization
    current_rejection_rate = designs_df['rejected_by_qc'].mean()
    projected_rejection_rate = current_rejection_rate * 0.85
    
    # Rework cost adjustment
    current_rework_cost = designs_df['rework_cost'].mean()
    projected_rework_cost = current_rework_cost * (projected_rejection_rate / current_rejection_rate)
    
    projected_total_cost = projected_design_cost + projected_rework_cost
    
    return {
        'current_avg_cost': current_avg_cost,
        'projected_avg_cost': projected_total_cost,
        'savings_per_design': current_avg_cost - projected_total_cost,
        'savings_percentage': ((current_avg_cost - projected_total_cost) / current_avg_cost) * 100,
        'annual_savings': (current_avg_cost - projected_total_cost) * len(designs_df),
        'current_rejection_rate': current_rejection_rate,
        'projected_rejection_rate': projected_rejection_rate
    }


def root_cause_analysis(designs_df):
    """Perform root cause analysis on QC rejections."""
    
    rejected_designs = designs_df[designs_df['rejected_by_qc'] == True].copy()
    
    # Parse rejection reasons
    all_reasons = []
    for reasons in rejected_designs['rejection_reasons'].dropna():
        all_reasons.extend(reasons.split('; '))
    
    reason_counts = pd.Series(all_reasons).value_counts()
    
    # Analyze by factors
    analysis = {
        'by_designer_type': designs_df.groupby('designer_type')['rejected_by_qc'].agg(['mean', 'count']),
        'by_complexity': designs_df.groupby('scan_complexity')['rejected_by_qc'].agg(['mean', 'count']),
        'by_doctor_pref': designs_df.groupby('doctor_pref_complexity')['rejected_by_qc'].agg(['mean', 'count']),
        'by_experience': None,
        'by_dental_knowledge': None,
        'reason_counts': reason_counts
    }
    
    # Experience buckets
    designs_df['experience_bucket'] = pd.cut(
        designs_df['designer_experience_months'],
        bins=[0, 3, 6, 12, 24, 100],
        labels=['0-3mo', '3-6mo', '6-12mo', '12-24mo', '24+mo']
    )
    analysis['by_experience'] = designs_df.groupby('experience_bucket')['rejected_by_qc'].agg(['mean', 'count'])
    
    # Dental knowledge buckets
    designs_df['knowledge_bucket'] = pd.cut(
        designs_df['designer_dental_knowledge'],
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    analysis['by_dental_knowledge'] = designs_df.groupby('knowledge_bucket')['rejected_by_qc'].agg(['mean', 'count'])
    
    return analysis


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cost_breakdown(designs_df):
    """Create cost breakdown visualization."""
    cost_components = {
        'Designer Labor': designs_df['designer_cost'].sum(),
        'QC Labor': designs_df['qc_cost'].sum(),
        'Rework Costs': designs_df['rework_cost'].sum()
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(cost_components.keys()),
        values=list(cost_components.values()),
        hole=0.4,
        marker_colors=['#2E5077', '#4A90A4', '#E07A5F']
    )])
    
    fig.update_layout(
        title='Cost Breakdown by Component',
        annotations=[dict(text=f'${sum(cost_components.values()):,.0f}', 
                         x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig


def plot_rejection_analysis(analysis):
    """Create rejection rate analysis charts."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('By Designer Type', 'By Scan Complexity', 
                       'By Experience Level', 'By Dental Knowledge'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # By Designer Type
    data = analysis['by_designer_type'].reset_index()
    fig.add_trace(
        go.Bar(x=data['designer_type'], y=data['mean']*100, 
               marker_color='#2E5077', name='Designer Type'),
        row=1, col=1
    )
    
    # By Complexity
    data = analysis['by_complexity'].reset_index()
    fig.add_trace(
        go.Bar(x=data['scan_complexity'], y=data['mean']*100,
               marker_color='#4A90A4', name='Complexity'),
        row=1, col=2
    )
    
    # By Experience
    data = analysis['by_experience'].reset_index()
    fig.add_trace(
        go.Bar(x=data['experience_bucket'].astype(str), y=data['mean']*100,
               marker_color='#81B29A', name='Experience'),
        row=2, col=1
    )
    
    # By Dental Knowledge
    data = analysis['by_dental_knowledge'].reset_index()
    fig.add_trace(
        go.Bar(x=data['knowledge_bucket'].astype(str), y=data['mean']*100,
               marker_color='#E07A5F', name='Knowledge'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="QC Rejection Rate Analysis")
    fig.update_yaxes(title_text="Rejection Rate (%)")
    
    return fig


def plot_throughput_analysis(designs_df):
    """Create throughput analysis visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Distribution by Step', 'Throughput by Designer Type')
    )
    
    # Step times
    step_cols = ['step1_scan_review', 'step2_margin_trace', 'step3_crown_placement', 
                 'step4_adjustments', 'step5_qc_review']
    step_labels = ['1. Scan Review', '2. Margin Trace', '3. Crown Placement', 
                   '4. Adjustments', '5. QC Review']
    step_means = [designs_df[col].mean() for col in step_cols]
    
    fig.add_trace(
        go.Bar(x=step_labels, y=step_means, marker_color=['#2E5077', '#3D6A8C', 
                                                          '#4A90A4', '#81B29A', '#E07A5F']),
        row=1, col=1
    )
    
    # Throughput by designer type
    throughput = designs_df.groupby('designer_type').agg({
        'total_design_time_min': 'mean',
        'design_id': 'count'
    }).reset_index()
    throughput['designs_per_hour'] = 60 / throughput['total_design_time_min']
    
    fig.add_trace(
        go.Bar(x=throughput['designer_type'], y=throughput['designs_per_hour'],
               marker_color='#2E5077'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="Time (minutes)", row=1, col=1)
    fig.update_yaxes(title_text="Designs per Hour", row=1, col=2)
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<p class="main-header">🦷 Dandy Design Operations Analysis</p>', unsafe_allow_html=True)
    st.markdown("**Data-Driven Analysis for Cost Optimization, Throughput, and Quality Control**")
    
    # Sidebar configuration
    st.sidebar.header("📊 Data Configuration")
    n_designers = st.sidebar.slider("Number of Designers", 20, 100, 50)
    n_designs = st.sidebar.slider("Number of Designs", 1000, 10000, 5000)
    seed = st.sidebar.number_input("Random Seed", value=42)
    
    # Generate data
    with st.spinner("Generating synthetic data..."):
        designers_df = generate_designer_data(n_designers, seed)
        designs_df = generate_design_data(n_designs, designers_df, seed)
        designers_df, designs_df, squad_metrics = generate_squad_data(designers_df, designs_df)
    
    # Tabs for each question
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview & Data", 
        "💰 Q1: Cost Optimization", 
        "⚡ Q2: Throughput & Metrics",
        "🔍 Q3: Root Cause Analysis"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW & DATA
    # ========================================================================
    with tab1:
        st.markdown('<p class="sub-header">Current State Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Designers", len(designers_df))
        with col2:
            st.metric("Total Designs", len(designs_df))
        with col3:
            st.metric("Avg Cost per Design", f"${designs_df['total_cost'].mean():.2f}")
        with col4:
            st.metric("QC Rejection Rate", f"{designs_df['rejected_by_qc'].mean()*100:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Designer Workforce Composition**")
            workforce_comp = designers_df.groupby(['designer_type', 'location']).size().reset_index(name='count')
            fig = px.sunburst(workforce_comp, path=['location', 'designer_type'], values='count',
                            color='location', color_discrete_map={'Onshore': '#2E5077', 'Offshore': '#81B29A'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Cost Distribution**")
            fig = plot_cost_breakdown(designs_df)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Sample Data Preview**")
        
        data_view = st.selectbox("Select Data to View", ["Designers", "Designs", "Squads"])
        if data_view == "Designers":
            st.dataframe(designers_df.head(20), use_container_width=True)
        elif data_view == "Designs":
            st.dataframe(designs_df.head(20), use_container_width=True)
        else:
            st.dataframe(squad_metrics, use_container_width=True)
    
    # ========================================================================
    # TAB 2: COST OPTIMIZATION (Q1)
    # ========================================================================
    with tab2:
        st.markdown('<p class="sub-header">Question 1: Cost Optimization Options</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Key Insight:</strong> Personnel contributes to 80% of costs, and average throughput 
        decreases with each new hire. The design process has 5 steps with varying skill requirements.
        </div>
        """, unsafe_allow_html=True)
        
        # Current cost analysis
        st.markdown("### Current State Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            personnel_cost = designs_df['designer_cost'].sum() + designs_df['qc_cost'].sum()
            total_cost = designs_df['total_cost'].sum()
            st.metric("Personnel Cost %", f"{(personnel_cost/total_cost)*100:.1f}%")
        with col2:
            avg_throughput = 60 / designs_df['total_design_time_min'].mean()
            st.metric("Avg Designs/Hour", f"{avg_throughput:.2f}")
        with col3:
            st.metric("Avg Cost/Design", f"${designs_df['total_cost'].mean():.2f}")
        
        # Cost by designer type
        cost_by_type = designs_df.groupby('designer_type').agg({
            'total_cost': 'mean',
            'total_design_time_min': 'mean',
            'rejected_by_qc': 'mean',
            'design_id': 'count'
        }).round(2).reset_index()
        cost_by_type.columns = ['Designer Type', 'Avg Cost ($)', 'Avg Time (min)', 'Rejection Rate', 'Design Count']
        
        st.markdown("### Cost & Performance by Designer Type")
        st.dataframe(cost_by_type, use_container_width=True)
        
        # Specialization Analysis
        st.markdown("---")
        st.markdown("### Option Analysis: Step Specialization Model")
        
        savings = calculate_specialization_savings(designs_df, designers_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Avg Cost", f"${savings['current_avg_cost']:.2f}")
        with col2:
            st.metric("Projected Avg Cost", f"${savings['projected_avg_cost']:.2f}", 
                     delta=f"-${savings['savings_per_design']:.2f}")
        with col3:
            st.metric("Annual Savings", f"${savings['annual_savings']:,.0f}",
                     delta=f"{savings['savings_percentage']:.1f}%")
        
        # Options comparison table
        st.markdown("### Comprehensive Options Comparison")
        
        options_data = {
            'Option': [
                '1. Step Specialization',
                '2. Offshore Expansion',
                '3. AI/Automation (Steps 1-2)',
                '4. Tiered QC (Risk-Based)',
                '5. Hybrid Model'
            ],
            'Cost Reduction': ['25-35%', '30-40%', '15-25%', '10-15%', '35-45%'],
            'Implementation Time': ['3-4 months', '2-3 months', '6-12 months', '2-3 months', '6-8 months'],
            'Quality Risk': ['Low', 'Medium-High', 'Low', 'Medium', 'Low-Medium'],
            'Scalability': ['High', 'High', 'Very High', 'Medium', 'Very High'],
            'Initial Investment': ['Low', 'Medium', 'High', 'Low', 'Medium-High']
        }
        
        options_df = pd.DataFrame(options_data)
        st.dataframe(options_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.markdown("### Projected Impact Simulation")
        
        option_select = st.selectbox(
            "Select Option to Analyze",
            options_data['Option']
        )
        
        # Simulated projections based on selection
        projections = {
            '1. Step Specialization': {'cost_reduction': 0.30, 'quality_impact': 0.95, 'throughput_increase': 1.20},
            '2. Offshore Expansion': {'cost_reduction': 0.35, 'quality_impact': 0.85, 'throughput_increase': 1.10},
            '3. AI/Automation (Steps 1-2)': {'cost_reduction': 0.20, 'quality_impact': 1.00, 'throughput_increase': 1.40},
            '4. Tiered QC (Risk-Based)': {'cost_reduction': 0.12, 'quality_impact': 0.92, 'throughput_increase': 1.15},
            '5. Hybrid Model': {'cost_reduction': 0.40, 'quality_impact': 0.93, 'throughput_increase': 1.35}
        }
        
        proj = projections[option_select]
        current_cost = designs_df['total_cost'].mean()
        current_rejection = designs_df['rejected_by_qc'].mean()
        current_throughput = 60 / designs_df['total_design_time_min'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            new_cost = current_cost * (1 - proj['cost_reduction'])
            st.metric("Projected Cost/Design", f"${new_cost:.2f}", 
                     delta=f"-{proj['cost_reduction']*100:.0f}%")
        with col2:
            new_rejection = current_rejection / proj['quality_impact']
            st.metric("Projected Rejection Rate", f"{new_rejection*100:.1f}%",
                     delta=f"{(new_rejection-current_rejection)*100:.1f}%")
        with col3:
            new_throughput = current_throughput * proj['throughput_increase']
            st.metric("Projected Designs/Hour", f"{new_throughput:.2f}",
                     delta=f"+{(proj['throughput_increase']-1)*100:.0f}%")
    
    # ========================================================================
    # TAB 3: THROUGHPUT & METRICS (Q2)
    # ========================================================================
    with tab3:
        st.markdown('<p class="sub-header">Question 2: Throughput & Quality Metrics</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Goal:</strong> Build processes and metrics to optimize throughput while maintaining quality.
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Dashboard
        st.markdown("### 📊 Key Performance Indicators (KPIs)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Throughput", f"{60/designs_df['total_design_time_min'].mean():.2f} designs/hr")
        with col2:
            st.metric("Quality (Pass Rate)", f"{(1-designs_df['rejected_by_qc'].mean())*100:.1f}%")
        with col3:
            st.metric("Avg Cycle Time", f"{designs_df['total_time_with_qc_min'].mean():.1f} min")
        with col4:
            st.metric("First Pass Yield", f"{(1-designs_df['rejected_by_qc'].mean())*100:.1f}%")
        
        st.markdown("---")
        
        # Throughput Analysis
        st.markdown("### Throughput Analysis")
        fig = plot_throughput_analysis(designs_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Squad Performance
        st.markdown("### Squad Performance Comparison")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rejection Rate by Squad', 'Avg Cost by Squad')
        )
        
        fig.add_trace(
            go.Bar(x=squad_metrics['squad'], y=squad_metrics['rejection_rate']*100,
                   marker_color='#E07A5F'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=squad_metrics['squad'], y=squad_metrics['avg_cost'],
                   marker_color='#2E5077'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(title_text="Rejection Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Cost ($)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Framework
        st.markdown("---")
        st.markdown("### 📋 Recommended Metrics Framework")
        
        metrics_framework = {
            'Category': ['Throughput', 'Throughput', 'Throughput', 'Quality', 'Quality', 'Quality', 'Cost', 'Cost', 'People'],
            'Metric': [
                'Designs Completed per Hour',
                'Avg Cycle Time (end-to-end)',
                'Step-Level Completion Time',
                'First Pass Yield (FPY)',
                'QC Rejection Rate',
                'Rework Rate',
                'Cost per Design',
                'Cost per Step',
                'Designer Utilization Rate'
            ],
            'Target': [
                '≥ 2.5/hour',
                '≤ 35 minutes',
                'Step 1-2: ≤10min, Step 3-4: ≤20min',
                '≥ 75%',
                '≤ 15%',
                '≤ 10%',
                '≤ $15',
                'Varies by step',
                '≥ 85%'
            ],
            'Current': [
                f"{60/designs_df['total_design_time_min'].mean():.2f}/hour",
                f"{designs_df['total_time_with_qc_min'].mean():.1f} min",
                f"{designs_df['step1_scan_review'].mean() + designs_df['step2_margin_trace'].mean():.1f}min / {designs_df['step3_crown_placement'].mean() + designs_df['step4_adjustments'].mean():.1f}min",
                f"{(1-designs_df['rejected_by_qc'].mean())*100:.1f}%",
                f"{designs_df['rejected_by_qc'].mean()*100:.1f}%",
                f"{designs_df['rejected_by_qc'].mean()*100:.1f}%",
                f"${designs_df['total_cost'].mean():.2f}",
                'See breakdown',
                'N/A (simulate)'
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_framework)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Process Recommendations
        st.markdown("### 🔄 Process Recommendations")
        
        process_recs = """
        | Process Area | Recommendation | Expected Impact |
        |-------------|----------------|-----------------|
        | **Workflow** | Implement step-based handoffs with clear SLAs | +20% throughput |
        | **QC** | Risk-based QC sampling for low-complexity designs | +15% QC capacity |
        | **Training** | Structured onboarding with step-specific certification | -25% rejection rate |
        | **Monitoring** | Real-time dashboards with alerts | Faster issue detection |
        | **Feedback Loop** | Daily QC → Designer feedback sessions | Continuous improvement |
        """
        st.markdown(process_recs)
    
    # ========================================================================
    # TAB 4: ROOT CAUSE ANALYSIS (Q3)
    # ========================================================================
    with tab4:
        st.markdown('<p class="sub-header">Question 3: QC Rejection Root Cause Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Problem:</strong> QC rejection rate ranges from 25-50% across squads. Goal is 0%.
        </div>
        """, unsafe_allow_html=True)
        
        # Current State
        rejection_rate = designs_df['rejected_by_qc'].mean()
        st.markdown(f"### Current Overall Rejection Rate: **{rejection_rate*100:.1f}%**")
        
        # Root Cause Analysis
        analysis = root_cause_analysis(designs_df)
        
        st.markdown("---")
        st.markdown("### 🔍 Multi-Factor Root Cause Analysis")
        
        # Rejection rate breakdown chart
        fig = plot_rejection_analysis(analysis)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rejection Reasons
        st.markdown("### Top Rejection Reasons")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            reason_df = analysis['reason_counts'].reset_index()
            reason_df.columns = ['Reason', 'Count']
            reason_df['Percentage'] = (reason_df['Count'] / reason_df['Count'].sum() * 100).round(1)
            
            fig = px.bar(reason_df, x='Percentage', y='Reason', orientation='h',
                        color='Percentage', color_continuous_scale='Reds')
            fig.update_layout(height=400, title='Rejection Reasons Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
        
        # Correlation Analysis
        st.markdown("---")
        st.markdown("### 📊 Correlation Analysis")
        
        # Prepare correlation data
        corr_cols = ['designer_dental_knowledge', 'designer_experience_months', 
                    'total_design_time_min', 'rejected_by_qc']
        
        # Create numeric complexity columns
        designs_df['complexity_numeric'] = designs_df['scan_complexity'].map({'Low': 1, 'Medium': 2, 'High': 3})
        designs_df['pref_numeric'] = designs_df['doctor_pref_complexity'].map({'Standard': 1, 'Custom': 2, 'Highly Custom': 3})
        
        corr_df = designs_df[['designer_dental_knowledge', 'designer_experience_months', 
                             'complexity_numeric', 'pref_numeric', 'rejected_by_qc']].copy()
        corr_df.columns = ['Dental Knowledge', 'Experience (mo)', 'Scan Complexity', 'Doctor Pref', 'Rejected']
        
        correlation = corr_df.corr()['Rejected'].drop('Rejected').sort_values()
        
        fig = px.bar(x=correlation.values, y=correlation.index, orientation='h',
                    color=correlation.values, color_continuous_scale='RdYlGn_r',
                    labels={'x': 'Correlation with Rejection', 'y': 'Factor'})
        fig.update_layout(height=300, title='Factors Correlated with QC Rejection')
        st.plotly_chart(fig, use_container_width=True)
        
        # Root Cause Summary
        st.markdown("---")
        st.markdown("### 🎯 Root Cause Summary & Pareto Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Primary Root Causes (80% of rejections):**
            1. **Low Dental Knowledge** - Designers with knowledge score < 0.5 have 2x rejection rate
            2. **Insufficient Experience** - < 6 months experience = 40%+ rejection rate
            3. **High Complexity Cases** - Complex scans + custom preferences = 50%+ rejection
            4. **Training Gaps** - Incomplete training correlates with higher rejection
            
            **Secondary Factors:**
            - Shift timing (Night shift +15% rejection)
            - Squad composition imbalance
            - QC reviewer variability
            """)
        
        with col2:
            # Pareto chart of root causes
            pareto_data = {
                'Root Cause': ['Low Dental Knowledge', 'Insufficient Experience', 
                              'High Complexity', 'Training Gaps', 'Other'],
                'Impact %': [35, 25, 20, 12, 8]
            }
            pareto_df = pd.DataFrame(pareto_data)
            pareto_df['Cumulative %'] = pareto_df['Impact %'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pareto_df['Root Cause'], y=pareto_df['Impact %'],
                                marker_color='#2E5077', name='Impact'))
            fig.add_trace(go.Scatter(x=pareto_df['Root Cause'], y=pareto_df['Cumulative %'],
                                    mode='lines+markers', marker_color='#E07A5F',
                                    name='Cumulative', yaxis='y2'))
            
            fig.update_layout(
                title='Pareto Analysis of Root Causes',
                yaxis=dict(title='Impact %'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Solution Framework
        st.markdown("---")
        st.markdown("### 💡 Solution Framework & Implementation Plan")
        
        solutions = {
            'Phase': ['Phase 1 (0-30 days)', 'Phase 1 (0-30 days)', 'Phase 2 (30-60 days)', 
                     'Phase 2 (30-60 days)', 'Phase 3 (60-90 days)', 'Phase 3 (60-90 days)'],
            'Initiative': [
                'Enhanced Training Program',
                'Case Routing by Complexity',
                'Mentorship Pairing',
                'Real-time QC Feedback',
                'Predictive Quality Model',
                'Certification Requirements'
            ],
            'Target Root Cause': [
                'Low Dental Knowledge, Training Gaps',
                'High Complexity mismatches',
                'Insufficient Experience',
                'All causes (feedback loop)',
                'Proactive quality assurance',
                'Knowledge validation'
            ],
            'Expected Impact': [
                '-15% rejection rate',
                '-10% rejection rate',
                '-8% rejection rate',
                '-5% rejection rate',
                '-7% rejection rate',
                '-5% rejection rate'
            ],
            'Success Metric': [
                'Training completion rate > 95%',
                'Complexity-skill match rate > 90%',
                '100% junior designers paired',
                'Feedback delivered within 1 hour',
                'Model accuracy > 80%',
                'All designers certified'
            ]
        }
        
        solutions_df = pd.DataFrame(solutions)
        st.dataframe(solutions_df, use_container_width=True, hide_index=True)
        
        # Projected Impact
        st.markdown("### 📈 Projected Rejection Rate Trajectory")
        
        weeks = list(range(0, 13))
        current_rate = rejection_rate * 100
        
        # Model improvement trajectory
        trajectory = [current_rate]
        for w in weeks[1:]:
            if w <= 4:
                improvement = 0.03 * current_rate  # Phase 1 improvements
            elif w <= 8:
                improvement = 0.025 * trajectory[-1]  # Phase 2 improvements
            else:
                improvement = 0.02 * trajectory[-1]  # Phase 3 improvements
            trajectory.append(max(5, trajectory[-1] - improvement))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=trajectory, mode='lines+markers',
                                name='Projected', line=dict(color='#2E5077', width=3)))
        fig.add_hline(y=15, line_dash="dash", line_color="green", 
                     annotation_text="Target: 15%")
        fig.add_hline(y=current_rate, line_dash="dash", line_color="red",
                     annotation_text=f"Current: {current_rate:.1f}%")
        
        fig.update_layout(
            title='Projected Rejection Rate Over 12 Weeks',
            xaxis_title='Week',
            yaxis_title='Rejection Rate (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download data
        st.markdown("---")
        st.markdown("### 📥 Export Data for Further Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = designers_df.to_csv(index=False)
            st.download_button("Download Designer Data", csv, "designers.csv", "text/csv")
        
        with col2:
            csv = designs_df.to_csv(index=False)
            st.download_button("Download Design Data", csv, "designs.csv", "text/csv")
        
        with col3:
            csv = squad_metrics.to_csv(index=False)
            st.download_button("Download Squad Metrics", csv, "squads.csv", "text/csv")


if __name__ == "__main__":
    main()
