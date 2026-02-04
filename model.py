"""
Dandy Design Operations Case Study - Interactive Analysis Tool
==============================================================
Interactive model with adjustable assumptions for cost, cycle time,
and comparison of optimization approaches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Dandy Design Operations Analysis",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.4rem; font-weight: 600; color: #2E5077; margin-top: 1.5rem; }
    .metric-big { font-size: 2.5rem; font-weight: 800; }
    .assumption-box { background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .insight-box { background-color: #E0F2FE; border-left: 4px solid #0284C7; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .success-box { background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .warning-box { background-color: #FEE2E2; border-left: 4px solid #EF4444; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .flow-step { display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">🦷 Dandy Design Operations Analysis</p>', unsafe_allow_html=True)
    st.markdown("**Interactive Model for Cost Optimization, Throughput & Quality**")
    
    # ========================================================================
    # SIDEBAR: MODEL INPUTS & ASSUMPTIONS
    # ========================================================================
    st.sidebar.header("📊 Model Inputs & Assumptions")
    st.sidebar.markdown("*Adjust these parameters to see impact on the model*")
    
    st.sidebar.subheader("⏱️ Cycle Time (minutes)")
    step1_time = st.sidebar.slider("Step 1: Scan Review", 5, 30, 15, help="Time to review and clean intraoral scan")
    step2_time = st.sidebar.slider("Step 2: Margin Trace", 10, 45, 25, help="Time to trace margin line on prepped tooth")
    step3_time = st.sidebar.slider("Step 3: Crown Placement", 20, 60, 40, help="Time to size and place initial crown")
    step4_time = st.sidebar.slider("Step 4: Adjustments", 30, 90, 60, help="Time for aesthetic and preference adjustments")
    step5_time = st.sidebar.slider("Step 5: QC Review", 20, 60, 40, help="Time for QC to review and revise")
    
    st.sidebar.subheader("💰 Hourly Rates ($)")
    st.sidebar.markdown("**Current State (Generalist)**")
    designer_rate = st.sidebar.slider("Designer (Steps 1-4)", 20, 50, 32, help="Hourly rate for designer doing all steps")
    qc_rate = st.sidebar.slider("QC Specialist", 35, 70, 50, help="Hourly rate for QC reviewer")
    
    st.sidebar.markdown("**Specialized Roles**")
    entry_rate = st.sidebar.slider("Entry-Level (Steps 1-2)", 10, 25, 14, help="Hourly rate for scan/margin specialists")
    mid_rate = st.sidebar.slider("Mid-Level (Step 3)", 18, 35, 24, help="Hourly rate for crown placement")
    senior_rate = st.sidebar.slider("Senior (Step 4)", 30, 55, 40, help="Hourly rate for adjustment specialists")
    
    st.sidebar.markdown("**Offshore Rates**")
    offshore_discount = st.sidebar.slider("Offshore Discount %", 30, 60, 45, help="Cost reduction for offshore labor")
    
    st.sidebar.subheader("📈 Other Parameters")
    designs_per_month = st.sidebar.number_input("Designs per Month", 1000, 20000, 8000)
    current_rejection_rate = st.sidebar.slider("Current QC Rejection Rate %", 20, 55, 35)
    rework_time_pct = st.sidebar.slider("Rework Time (% of original)", 30, 70, 50)
    
    # ========================================================================
    # CALCULATIONS
    # ========================================================================
    
    # Current state calculations
    total_design_time = step1_time + step2_time + step3_time + step4_time
    total_time_with_qc = total_design_time + step5_time
    
    # Cost calculations (current state)
    designer_cost = (total_design_time / 60) * designer_rate
    qc_cost = (step5_time / 60) * qc_rate
    avg_rework_time = total_design_time * (rework_time_pct / 100) * (current_rejection_rate / 100)
    rework_cost = (avg_rework_time / 60) * designer_rate
    current_total_cost = designer_cost + qc_cost + rework_cost
    
    # Specialized model calculations
    steps12_time = step1_time + step2_time
    step3_time_val = step3_time
    step4_time_val = step4_time
    
    spec_cost_steps12 = (steps12_time / 60) * entry_rate
    spec_cost_step3 = (step3_time_val / 60) * mid_rate
    spec_cost_step4 = (step4_time_val / 60) * senior_rate
    spec_qc_cost = (step5_time / 60) * qc_rate
    
    # Assume 20% reduction in rejection rate due to specialization
    spec_rejection_rate = current_rejection_rate * 0.80
    spec_rework_cost = (avg_rework_time * 0.80 / 60) * ((entry_rate + mid_rate + senior_rate) / 3)
    specialized_total_cost = spec_cost_steps12 + spec_cost_step3 + spec_cost_step4 + spec_qc_cost + spec_rework_cost
    
    # Offshore model calculations
    offshore_rate_multiplier = (100 - offshore_discount) / 100
    offshore_designer_rate = designer_rate * offshore_rate_multiplier
    offshore_qc_rate = qc_rate * offshore_rate_multiplier
    
    offshore_designer_cost = (total_design_time / 60) * offshore_designer_rate
    offshore_qc_cost = (step5_time / 60) * offshore_qc_rate
    # Assume 15% increase in rejection rate for offshore
    offshore_rejection_rate = min(current_rejection_rate * 1.15, 60)
    offshore_rework_cost = (total_design_time * (rework_time_pct / 100) * (offshore_rejection_rate / 100) / 60) * offshore_designer_rate
    offshore_total_cost = offshore_designer_cost + offshore_qc_cost + offshore_rework_cost
    
    # AI/Automation model calculations (Steps 1-2 automated)
    ai_time_reduction = 0.7  # AI does steps 1-2 in 70% less time
    ai_steps12_time = steps12_time * (1 - ai_time_reduction)
    ai_steps12_cost = 0.50  # Fixed cost per design for AI processing
    
    ai_designer_cost = ((step3_time_val + step4_time_val) / 60) * designer_rate
    ai_qc_cost = (step5_time / 60) * qc_rate
    # Assume consistent quality from AI
    ai_rejection_rate = current_rejection_rate * 0.90
    ai_rework_cost = ((step3_time_val + step4_time_val) * (rework_time_pct / 100) * (ai_rejection_rate / 100) / 60) * designer_rate
    ai_total_cost = ai_steps12_cost + ai_designer_cost + ai_qc_cost + ai_rework_cost
    
    # Throughput calculations
    current_throughput = 60 / total_time_with_qc
    # Specialized: parallel processing can improve throughput
    specialized_throughput = current_throughput * 1.25
    offshore_throughput = current_throughput * 1.10  # Timezone advantage
    ai_throughput = 60 / (ai_steps12_time + step3_time_val + step4_time_val + step5_time)
    
    # Time savings calculations
    current_time_per_design = total_time_with_qc + avg_rework_time
    spec_time_per_design = total_time_with_qc + (avg_rework_time * 0.80)
    ai_time_per_design = ai_steps12_time + step3_time_val + step4_time_val + step5_time + (avg_rework_time * 0.90)
    
    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2, tab3 = st.tabs([
        "💰 Q1: Cost Optimization Model", 
        "⚡ Q2: Processes & Metrics",
        "🔍 Q3: Root Cause & Pilots"
    ])
    
    # ========================================================================
    # TAB 1: COST OPTIMIZATION MODEL
    # ========================================================================
    with tab1:
        st.markdown('<p class="sub-header">Question 1: Cost Optimization Options</p>', unsafe_allow_html=True)
        
        # Model Assumptions Display
        st.markdown("### 📋 Model Inputs & Assumptions")
        st.markdown("""
        <div class="assumption-box">
        <strong>⚠️ Key Assumptions (adjustable in sidebar):</strong><br>
        These inputs drive all calculations. Adjust them to test different scenarios and see how results change.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏱️ Cycle Time Assumptions**")
            cycle_df = pd.DataFrame({
                'Step': ['1. Scan Review', '2. Margin Trace', '3. Crown Placement', '4. Adjustments', '5. QC Review'],
                'Time (min)': [step1_time, step2_time, step3_time, step4_time, step5_time],
                'Dental Knowledge': ['Not Required', 'Not Required', 'Helpful', 'Required', 'Required']
            })
            st.dataframe(cycle_df, hide_index=True, use_container_width=True)
            st.metric("Total Cycle Time", f"{total_time_with_qc} min ({total_time_with_qc/60:.1f} hrs)")
        
        with col2:
            st.markdown("**💰 Labor Cost Assumptions**")
            rate_df = pd.DataFrame({
                'Role': ['Generalist Designer', 'QC Specialist', 'Entry-Level', 'Mid-Level', 'Senior Specialist'],
                'Hourly Rate': [f'${designer_rate}', f'${qc_rate}', f'${entry_rate}', f'${mid_rate}', f'${senior_rate}']
            })
            st.dataframe(rate_df, hide_index=True, use_container_width=True)
        
        with col3:
            st.markdown("**📊 Operational Assumptions**")
            st.metric("Monthly Volume", f"{designs_per_month:,} designs")
            st.metric("QC Rejection Rate", f"{current_rejection_rate}%")
            st.metric("Avg Rework Time", f"{rework_time_pct}% of original")
        
        st.markdown("---")
        
        # Current State Flow
        st.markdown("### 🔄 Current State: Process Flow & Economics")
        
        # Flow diagram using columns
        st.markdown("**Current Process Flow (Single Designer Model)**")
        
        flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
        with flow_col1:
            st.markdown(f"""
            <div style="background: #3B82F6; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 1</strong><br>
                Scan Review<br>
                <span style="font-size: 20px; font-weight: 800;">{step1_time} min</span><br>
                <small>${designer_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with flow_col2:
            st.markdown(f"""
            <div style="background: #3B82F6; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 2</strong><br>
                Margin Trace<br>
                <span style="font-size: 20px; font-weight: 800;">{step2_time} min</span><br>
                <small>${designer_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with flow_col3:
            st.markdown(f"""
            <div style="background: #8B5CF6; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 3</strong><br>
                Crown Place<br>
                <span style="font-size: 20px; font-weight: 800;">{step3_time} min</span><br>
                <small>${designer_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with flow_col4:
            st.markdown(f"""
            <div style="background: #EC4899; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 4</strong><br>
                Adjustments<br>
                <span style="font-size: 20px; font-weight: 800;">{step4_time} min</span><br>
                <small>${designer_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with flow_col5:
            st.markdown(f"""
            <div style="background: #EF4444; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 5</strong><br>
                QC Review<br>
                <span style="font-size: 20px; font-weight: 800;">{step5_time} min</span><br>
                <small>${qc_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<p style='text-align: center; color: #64748b;'>⬇️ {current_rejection_rate}% rejection rate → Rework loop adds ~{avg_rework_time:.1f} min average per design</p>", unsafe_allow_html=True)
        
        # Current state metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Designer Cost", f"${designer_cost:.2f}")
        with col2:
            st.metric("QC Cost", f"${qc_cost:.2f}")
        with col3:
            st.metric("Avg Rework Cost", f"${rework_cost:.2f}")
        with col4:
            st.metric("**Total Cost/Design**", f"${current_total_cost:.2f}", delta=None)
        
        st.markdown("---")
        
        # OPTION 1: Step Specialization
        st.markdown("### 🎯 Option 1: Step Specialization Model")
        
        st.markdown("**Optimized Process Flow (Specialized Roles)**")
        
        spec_col1, spec_col2, spec_col3, spec_col4, spec_col5 = st.columns(5)
        with spec_col1:
            st.markdown(f"""
            <div style="background: #10B981; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 1</strong><br>
                Scan Review<br>
                <span style="font-size: 20px; font-weight: 800;">{step1_time} min</span><br>
                <small style="background: #065F46; padding: 2px 6px; border-radius: 4px;">${entry_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with spec_col2:
            st.markdown(f"""
            <div style="background: #10B981; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 2</strong><br>
                Margin Trace<br>
                <span style="font-size: 20px; font-weight: 800;">{step2_time} min</span><br>
                <small style="background: #065F46; padding: 2px 6px; border-radius: 4px;">${entry_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with spec_col3:
            st.markdown(f"""
            <div style="background: #F59E0B; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 3</strong><br>
                Crown Place<br>
                <span style="font-size: 20px; font-weight: 800;">{step3_time} min</span><br>
                <small style="background: #B45309; padding: 2px 6px; border-radius: 4px;">${mid_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with spec_col4:
            st.markdown(f"""
            <div style="background: #EC4899; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 4</strong><br>
                Adjustments<br>
                <span style="font-size: 20px; font-weight: 800;">{step4_time} min</span><br>
                <small style="background: #9D174D; padding: 2px 6px; border-radius: 4px;">${senior_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        with spec_col5:
            st.markdown(f"""
            <div style="background: #EF4444; color: white; padding: 15px; border-radius: 8px; text-align: center;">
                <strong>Step 5</strong><br>
                QC Review<br>
                <span style="font-size: 20px; font-weight: 800;">{step5_time} min</span><br>
                <small style="background: #B91C1C; padding: 2px 6px; border-radius: 4px;">${qc_rate}/hr</small>
            </div>
            """, unsafe_allow_html=True)
        
        spec_savings = current_total_cost - specialized_total_cost
        spec_savings_pct = (spec_savings / current_total_cost) * 100
        time_savings_spec = current_time_per_design - spec_time_per_design
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Steps 1-2 Cost", f"${spec_cost_steps12:.2f}", delta=f"-${(steps12_time/60)*designer_rate - spec_cost_steps12:.2f}")
        with col2:
            st.metric("Step 3 Cost", f"${spec_cost_step3:.2f}")
        with col3:
            st.metric("Step 4 Cost", f"${spec_cost_step4:.2f}")
        with col4:
            st.metric("**Total Cost/Design**", f"${specialized_total_cost:.2f}", delta=f"-${spec_savings:.2f} ({spec_savings_pct:.1f}%)")
        
        st.markdown("---")
        
        # OPTION 2: Offshore Expansion
        st.markdown("### 🌏 Option 2: Offshore Expansion")
        
        offshore_savings = current_total_cost - offshore_total_cost
        offshore_savings_pct = (offshore_savings / current_total_cost) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            **Model:** Same process, offshore labor at {offshore_discount}% lower cost
            - Designer Rate: ${offshore_designer_rate:.2f}/hr (vs ${designer_rate})
            - QC Rate: ${offshore_qc_rate:.2f}/hr (vs ${qc_rate})
            """)
        with col2:
            st.metric("Total Cost/Design", f"${offshore_total_cost:.2f}", delta=f"-${offshore_savings:.2f} ({offshore_savings_pct:.1f}%)")
        with col3:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Quality Risk:</strong><br>
            Model assumes {((offshore_rejection_rate/current_rejection_rate)-1)*100:.0f}% higher rejection rate due to communication/timezone challenges. This assumption should be validated.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # OPTION 3: AI/Automation
        st.markdown("### 🤖 Option 3: AI/Automation (Steps 1-2)")
        
        ai_savings = current_total_cost - ai_total_cost
        ai_savings_pct = (ai_savings / current_total_cost) * 100
        ai_time_savings = current_time_per_design - ai_time_per_design
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            **Model:** AI handles Steps 1-2 automatically
            - AI Processing: ~{ai_steps12_time:.1f} min (vs {steps12_time} min manual)
            - Fixed cost: ${ai_steps12_cost:.2f}/design for AI
            - Human handles Steps 3-5
            """)
        with col2:
            st.metric("Total Cost/Design", f"${ai_total_cost:.2f}", delta=f"-${ai_savings:.2f} ({ai_savings_pct:.1f}%)")
            st.metric("Time Savings/Design", f"{ai_time_savings:.1f} min")
        with col3:
            st.markdown(f"""
            <div class="insight-box">
            <strong>💡 Note:</strong><br>
            Requires 6-12 month development. High upfront investment but highly scalable. AI cost assumption of ${ai_steps12_cost}/design is based on current ML processing costs.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # COMPARISON SUMMARY
        st.markdown("### 📊 Options Comparison Summary")
        
        comparison_data = {
            'Approach': ['Current State', '1. Step Specialization', '2. Offshore Expansion', '3. AI/Automation'],
            'Cost/Design': [f'${current_total_cost:.2f}', f'${specialized_total_cost:.2f}', f'${offshore_total_cost:.2f}', f'${ai_total_cost:.2f}'],
            'Savings %': ['—', f'{spec_savings_pct:.1f}%', f'{offshore_savings_pct:.1f}%', f'{ai_savings_pct:.1f}%'],
            'Monthly Savings': ['—', f'${spec_savings * designs_per_month:,.0f}', f'${offshore_savings * designs_per_month:,.0f}', f'${ai_savings * designs_per_month:,.0f}'],
            'Throughput': [f'{current_throughput:.2f}/hr', f'{specialized_throughput:.2f}/hr', f'{offshore_throughput:.2f}/hr', f'{ai_throughput:.2f}/hr'],
            'Implementation': ['—', '3-4 months', '2-3 months', '6-12 months'],
            'Quality Risk': ['Baseline', 'Low', 'Medium-High', 'Low']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # Visual comparison chart
        fig = go.Figure()
        
        approaches = ['Current', 'Specialization', 'Offshore', 'AI/Automation']
        costs = [current_total_cost, specialized_total_cost, offshore_total_cost, ai_total_cost]
        colors = ['#64748B', '#10B981', '#F59E0B', '#8B5CF6']
        
        fig.add_trace(go.Bar(
            x=approaches,
            y=costs,
            marker_color=colors,
            text=[f'${c:.2f}' for c in costs],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Cost per Design Comparison (Based on Model Assumptions)',
            yaxis_title='Cost ($)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="assumption-box">
        <strong>⚠️ Important:</strong> These projections are based on the assumptions in the sidebar. 
        Actual results will depend on real-world implementation factors. Use this model to understand 
        directional impact and sensitivity to different assumptions.
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: PROCESSES & METRICS
    # ========================================================================
    with tab2:
        st.markdown('<p class="sub-header">Question 2: Processes & Metrics Framework</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Goal:</strong> Optimize throughput while maintaining quality. Build processes and metrics to drive towards these goals.
        </div>
        """, unsafe_allow_html=True)
        
        # KPI Framework
        st.markdown("### 📊 Key Performance Indicators (KPIs)")
        
        st.markdown("""
        <div class="assumption-box">
        <strong>📖 First Pass Yield (FPY) Explained:</strong><br><br>
        First Pass Yield is the percentage of designs that pass QC review on the <em>first attempt</em> without requiring rework. 
        <br><br>
        <strong>Formula:</strong> FPY = (Designs Passed on First Review) / (Total Designs Submitted to QC) × 100%
        <br><br>
        <strong>Example:</strong> If 100 designs are submitted to QC and 65 pass on first review, FPY = 65%
        <br><br>
        <em>Why it matters:</em> Higher FPY means less rework, lower costs, and faster throughput. It's a leading indicator of design quality and directly impacts cost per design.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate current FPY from rejection rate
        current_fpy = 100 - current_rejection_rate
        
        kpi_data = {
            'Category': ['Throughput', 'Throughput', 'Quality', 'Quality', 'Cost', 'People'],
            'Metric': [
                'Designs per Day',
                'Avg Cycle Time',
                'First Pass Yield (FPY)',
                'QC Rejection Rate',
                'Cost per Design',
                'Designer Utilization'
            ],
            'Definition': [
                'Number of designs completed per designer per day',
                'End-to-end time from scan receipt to QC approval',
                '% of designs passing QC on first submission',
                '% of designs rejected by QC (= 100% - FPY)',
                'Total labor cost to produce one approved design',
                '% of available work time spent on productive design work'
            ],
            'Target': ['≥ 2.5', '≤ 3.5 hrs', '≥ 75%', '≤ 25%', '≤ $100', '≥ 85%'],
            'Current (Model)': [
                f'{current_throughput:.2f}',
                f'{total_time_with_qc/60:.1f} hrs',
                f'{current_fpy:.0f}%',
                f'{current_rejection_rate}%',
                f'${current_total_cost:.2f}',
                '~78% (assumed)'
            ]
        }
        
        kpi_df = pd.DataFrame(kpi_data)
        st.dataframe(kpi_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Process Framework
        st.markdown("### 🔄 Process Framework")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 1. Real-Time Monitoring")
            st.markdown("""
            - **Dashboard**: Live throughput, queue depth, cycle times by step
            - **Alerts**: Automatic notification when metrics fall below threshold
            - **Visibility**: Squad-level and individual-level views
            
            *Purpose: Early detection of issues, data-driven decision making*
            """)
            
            st.markdown("#### 🔁 2. Feedback Loops")
            st.markdown("""
            - **Immediate**: QC provides rejection reason at time of review
            - **Daily**: Stand-up reviews common issues from previous day
            - **Weekly**: Squad reviews aggregate patterns and trends
            
            *Purpose: Continuous learning and rapid improvement*
            """)
        
        with col2:
            st.markdown("#### 📋 3. Structured Workflow")
            st.markdown("""
            - **SLAs**: Target time per step (see model inputs)
            - **Routing**: Match case complexity to designer experience
            - **Handoffs**: Clear protocols for step transitions
            
            *Purpose: Predictable output, reduced variability*
            """)
            
            st.markdown("#### 📚 4. Training & Certification")
            st.markdown("""
            - **Onboarding**: Structured curriculum by role/step
            - **Certification**: Skill validation before independent work
            - **Mentorship**: Pair junior with senior designers
            
            *Purpose: Consistent quality, career development*
            """)
        
        st.markdown("---")
        
        # Measurement Hierarchy
        st.markdown("### 📈 Measurement & Review Cadence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Metric Hierarchy")
            hierarchy_df = pd.DataFrame({
                'Level': ['Organization', 'Squad', 'Individual'],
                'Metrics': [
                    'Total cost/design, aggregate throughput, overall FPY',
                    'Squad rejection rate, squad cycle time, squad utilization',
                    'Personal FPY, designs/day, personal avg cycle time'
                ],
                'Owner': ['Ops Leadership', 'People Manager', 'Designer + QC Lead']
            })
            st.dataframe(hierarchy_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Review Cadence")
            cadence_df = pd.DataFrame({
                'Frequency': ['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                'Review Type': [
                    'Stand-up: yesterday issues, today priorities',
                    'Squad Review: KPI trends, improvement actions',
                    'Cross-Squad: benchmarking, best practice sharing',
                    'Strategic: process changes, investment decisions'
                ]
            })
            st.dataframe(cadence_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Data Infrastructure
        st.markdown("### 🗄️ Data Infrastructure Requirements")
        
        data_req = pd.DataFrame({
            'Data Point': [
                'Step timestamps',
                'Designer ID per step',
                'QC decision + reason',
                'Case attributes',
                'Rework iterations'
            ],
            'Capture Method': [
                'Software event logging',
                'Software assignment tracking',
                'Structured dropdown at review',
                'Intake form (complexity, doctor preferences)',
                'Software version tracking'
            ],
            'Use': [
                'Cycle time calculation, bottleneck identification',
                'Individual performance, workload distribution',
                'Root cause analysis, training needs identification',
                'Case routing, performance normalization',
                'True cost calculation, quality trending'
            ]
        })
        st.dataframe(data_req, hide_index=True, use_container_width=True)
    
    # ========================================================================
    # TAB 3: ROOT CAUSE & PILOTS
    # ========================================================================
    with tab3:
        st.markdown('<p class="sub-header">Question 3: Root Cause Analysis & Solving</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Problem Statement:</strong> QC rejection rate ranges from 25-50% across squads. Goal is 0%.
        <br><br>
        <strong>Approach:</strong> Use a data-driven hypothesis tree to systematically identify root causes, then design pilots to validate and solve.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Hypothesis Tree Methodology
        st.markdown("### 🌳 Hypothesis Tree Approach")
        
        st.markdown("""
        A hypothesis tree is a structured framework for problem-solving that:
        1. Breaks a complex problem into mutually exclusive, collectively exhaustive (MECE) components
        2. Forms testable hypotheses at each branch
        3. Identifies specific data needed to prove/disprove each hypothesis
        
        **For each branch, we ask: "What data would prove or disprove this hypothesis?"**
        """)
        
        st.markdown("#### Level 0: Problem Statement")
        st.markdown("""
        <div style="background: #EF4444; color: white; padding: 20px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: 600;">
        Why is QC rejection rate 25-50% across squads?
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Level 1: Primary Hypotheses (MECE)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #3B82F6; color: white; padding: 20px; border-radius: 8px; min-height: 120px;">
            <strong style="font-size: 16px;">H1: People Problem</strong><br><br>
            Designers lack the skills, knowledge, or experience to consistently produce quality work
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #8B5CF6; color: white; padding: 20px; border-radius: 8px; min-height: 120px;">
            <strong style="font-size: 16px;">H2: Process Problem</strong><br><br>
            Workflow, tools, or working conditions create conditions that lead to errors
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #EC4899; color: white; padding: 20px; border-radius: 8px; min-height: 120px;">
            <strong style="font-size: 16px;">H3: Input Problem</strong><br><br>
            Case complexity, requirements, or inputs are mismatched with capabilities
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Level 2: Sub-Hypotheses & Data Tests")
        
        hypothesis_tree = {
            'Primary': [
                'H1: People', 'H1: People', 'H1: People',
                'H2: Process', 'H2: Process', 'H2: Process',
                'H3: Input', 'H3: Input'
            ],
            'Sub-Hypothesis': [
                'H1a: Insufficient dental knowledge for steps 3-4',
                'H1b: Inadequate training or onboarding',
                'H1c: Inconsistent QC standards across reviewers',
                'H2a: Time pressure leads to cutting corners',
                'H2b: Poor handoffs cause information loss',
                'H2c: Software/tool limitations cause errors',
                'H3a: Complex cases misassigned to junior designers',
                'H3b: Doctor preferences unclear or changing'
            ],
            'Data Test to Validate': [
                'Correlate rejection rate with dental knowledge assessment score',
                'Compare rejection rate by tenure bucket and training completion status',
                'Compare rejection rates for same designs reviewed by different QCs',
                'Correlate rejection rate with queue depth and utilization at time of design',
                'Compare rejection rate when same designer does all steps vs multiple designers',
                'Categorize rejections as software-limitation vs skill-gap related',
                'Cross-tab rejection rate by case complexity × designer experience level',
                'Compare rejection rate for standard vs custom preference cases'
            ],
            'If True, We Expect': [
                'Strong negative correlation (lower score = higher rejection)',
                'Newer and untrained designers have significantly higher rejection rates',
                'High variance in rejection decisions for identical designs',
                'Positive correlation (higher load = more rejections)',
                'Higher rejection when work is split across designers',
                'Significant % of rejections have software root cause',
                'Interaction effect: complex + junior = very high rejection',
                'Custom preference cases have significantly higher rejection'
            ]
        }
        
        hypothesis_df = pd.DataFrame(hypothesis_tree)
        st.dataframe(hypothesis_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Root Cause Investigation Process
        st.markdown("### 🔍 Root Cause Investigation Process")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Step 1: Data Collection** (Week 1-2)
            - Extract 3 months of rejection data with reason codes
            - Join with designer attributes (tenure, training, assessments)
            - Join with case attributes (complexity, doctor, preferences)
            - Join with operational data (queue depth, time of day, shift)
            """)
            
            st.markdown("""
            **Step 2: Quantitative Analysis** (Week 2-3)
            - Run correlation analysis for each sub-hypothesis
            - Segment rejection rate by each factor independently
            - Look for interaction effects (e.g., complexity × experience)
            - Statistical significance testing to validate relationships
            """)
        
        with col2:
            st.markdown("""
            **Step 3: Qualitative Validation** (Week 3-4)
            - Interview 5-10 designers (mix of high and low rejection)
            - Shadow 2-3 QC reviewers during review sessions
            - Review sample of 20 rejected designs with SMEs
            - Identify root causes not visible in quantitative data
            """)
            
            st.markdown("""
            **Step 4: Prioritization** (Week 4)
            - Rank hypotheses by % of rejections explained
            - Assess feasibility and cost of addressing each root cause
            - Select top 2-3 root causes for pilot interventions
            """)
        
        st.markdown("---")
        
        # Pilot Experiments
        st.markdown("### 🧪 Pilot Experiments to Validate & Solve")
        
        st.markdown("""
        <div class="insight-box">
        <strong>Pilot Philosophy:</strong> Test solutions with small groups before committing to organization-wide rollout. 
        This allows us to validate impact, learn what works, and iterate quickly with minimal risk.
        </div>
        """, unsafe_allow_html=True)
        
        pilot_data = {
            'Pilot': [
                '🎓 A: Targeted Training',
                '🔀 B: Complexity Routing',
                '⚡ C: Real-Time Feedback',
                '👥 D: Mentorship Pairing',
                '📏 E: QC Calibration'
            ],
            'Tests Hypothesis': [
                'H1a/H1b: Knowledge and training gaps',
                'H3a: Complexity-skill mismatch',
                'H2a/H2b: Process & feedback gaps',
                'H1b: Experience gap',
                'H1c: Inconsistent QC standards'
            ],
            'Pilot Design': [
                'Select 10 high-rejection designers. Provide 2-week intensive training on top 3 rejection reasons. Compare their before/after rejection rate.',
                'For 2 weeks, route high-complexity cases ONLY to designers with >12 mo experience. Compare rejection rate vs control (random assignment).',
                'One squad gets QC feedback within 1 hour of rejection. Control squad gets end-of-day batch feedback. Compare learning curves.',
                'Pair 5 junior designers (<6 mo) with senior mentors (weekly 1:1s, design review). Compare rejection trajectory vs unpaired juniors.',
                'All QCs review same 20 designs blind. Measure inter-rater agreement. If <80%, implement weekly calibration sessions.'
            ],
            'Success Metric': [
                '≥30% reduction in rejection rate for pilot group',
                '≥20% lower rejection rate for complex cases vs control',
                'Faster FPY improvement (steeper learning curve)',
                'Faster time-to-competency (weeks to reach 75% FPY)',
                '≥80% agreement between QCs on same designs'
            ],
            'Duration': ['4 weeks', '2 weeks', '4 weeks', '6 weeks', '2 weeks'],
            'Sample': ['10 designers', '100 cases each arm', '2 squads', '5 pairs (10 people)', 'All QCs']
        }
        
        pilot_df = pd.DataFrame(pilot_data)
        st.dataframe(pilot_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Implementation Roadmap
        st.markdown("### 📋 Implementation Roadmap")
        
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 25px;">
        
        **Phase 1: Diagnose (Weeks 1-4)**
        - Week 1-2: Data collection and preparation
        - Week 2-3: Quantitative hypothesis testing
        - Week 3-4: Qualitative validation and prioritization
        - Output: Ranked list of validated root causes
        
        **Phase 2: Pilot (Weeks 5-10)**
        - Week 5: Design and launch 2-3 pilots targeting top root causes
        - Week 6-9: Run pilots with weekly monitoring
        - Week 10: Evaluate results against success metrics
        - Output: Validated solutions with measured impact
        
        **Phase 3: Scale (Weeks 11-16)**
        - Week 11-12: Develop rollout plan for successful pilots
        - Week 13-16: Phased rollout across organization
        - Output: Organization-wide implementation
        
        **Phase 4: Sustain (Ongoing)**
        - Integrate successful interventions into standard operating procedures
        - Continuous monitoring of rejection rate metrics
        - Quarterly review cycle for emerging issues
        
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Key Principles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #ECFDF5; padding: 20px; border-radius: 8px; border-left: 4px solid #10B981;">
            <strong>📊 Data-Driven</strong><br><br>
            Let data guide hypothesis prioritization. Measure before and after every intervention. Be willing to abandon hypotheses that data doesn't support.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #E0F2FE; padding: 20px; border-radius: 8px; border-left: 4px solid #0284C7;">
            <strong>🧪 Pilot First</strong><br><br>
            Test solutions with small groups before scaling. Validate impact before committing resources. Learn fast, fail fast, iterate.
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #FEF3C7; padding: 20px; border-radius: 8px; border-left: 4px solid #F59E0B;">
            <strong>🔄 Continuous</strong><br><br>
            Root cause analysis is not a one-time exercise. Build ongoing measurement and feedback loops. New issues will emerge as old ones are solved.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
