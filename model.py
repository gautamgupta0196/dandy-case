
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys

st.set_page_config(page_title="WTC Transit System Rebuild - Case Analysis", layout="wide")

try:
    st.title("🚇 NYC Transit System Rebuild After 9/11")
    st.markdown("""
    **Case Study Analysis: The Aftermath of the World Trade Center Collapse**
    
    This dashboard analyzes the project management strategies used to rebuild the 1/9 subway line 
    in just 7 months, when initial estimates projected 2 years.
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Project Cost", "$162.5M", help="Including $3M bonus")
    with col2:
        st.metric("Completion Time", "7 Months", "-17 Months", delta_color="inverse", help="vs. initial 2-year estimate")
    with col3:
        st.metric("Workers", "350", help="Working 24/7")
    with col4:
        st.metric("Bonus Earned", "$3M", help="$100K/day × 30 days early")
    
    st.divider()
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Project Timeline", "🏗️ Construction Phases", "📈 Performance Metrics", "💡 Key Learnings"])
    
    with tab1:
        st.subheader("Project Schedule Timeline")
        
        # Project milestones data - simplified to avoid overlapping dates
        milestones = pd.DataFrame({
            'Milestone': [
                'Project Start',
                '21 Days Behind\n+ Incentive Signed',
                'June 30 Status\n(17 days behind)',
                'July 31 Status\n(7 days behind)',
                'Geometry Car Test',
                'Third Rail Energized',
                'Test Train Run',
                'Service Opens',
                'Original Deadline'
            ],
            'Short_Label': [
                'Start',
                'Incentive',
                'Jun Status',
                'Jul Status',
                'Geo Test',
                '3rd Rail',
                'Test Run',
                'Opens!',
                'Deadline'
            ],
            'Date': [
                '2002-02-01', '2002-05-10', '2002-06-30',
                '2002-07-31', '2002-08-15', '2002-08-31', '2002-08-30',
                '2002-09-15', '2002-09-30'
            ],
            'Status': [
                'Start', 'Action', 'Progress', 
                'Progress', 'Testing', 'Testing', 'Testing',
                'Complete', 'Deadline'
            ],
            'Days_Behind': [0, 21, 17, 7, 0, 0, 0, 0, -15]
        })
        milestones['Date'] = pd.to_datetime(milestones['Date'])
        
        # Timeline chart using go.Figure for better control
        color_map = {
            'Start': '#2E86AB', 'Behind': '#E63946', 'Action': '#F4A261',
            'Progress': '#2A9D8F', 'Testing': '#9B5DE5', 'Complete': '#00A676', 'Deadline': '#6C757D'
        }
        
        fig_timeline = go.Figure()
        
        # Add scatter points by status
        for status in milestones['Status'].unique():
            df_status = milestones[milestones['Status'] == status]
            fig_timeline.add_trace(go.Scatter(
                x=df_status['Date'],
                y=df_status['Days_Behind'],
                mode='markers',
                marker=dict(size=14, color=color_map.get(status, '#333')),
                name=status,
                hovertemplate='<b>%{customdata[0]}</b><br>Date: %{x|%b %d, %Y}<br>Days Behind: %{y}<extra></extra>',
                customdata=df_status[['Milestone']]
            ))
        
        # Add annotations with offset positions to prevent overlap
        annotations = []
        y_offsets = [25, 35, 25, 25, -30, -45, 25, -30, 25]  # Custom offsets for each point
        x_offsets = [0, 0, 0, 0, 0, 0, -15, 0, 0]
        
        for i, row in milestones.iterrows():
            annotations.append(dict(
                x=row['Date'],
                y=row['Days_Behind'],
                text=row['Short_Label'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='#666',
                ax=x_offsets[i],
                ay=y_offsets[i],
                font=dict(size=10, color='#333'),
                bgcolor='rgba(255,255,255,0.8)',
                borderpad=2
            ))
        
        fig_timeline.update_layout(
            title='Project Schedule Performance Over Time',
            template='plotly_white',
            margin=dict(l=40, r=40, t=80, b=100),
            yaxis_title='Days Behind Schedule (negative = ahead)',
            xaxis_title='Date (2002)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            height=500,
            annotations=annotations
        )
        
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="green", 
                               annotation_text="On Schedule", annotation_position="right")
        
        st.plotly_chart(fig_timeline)
        
        # Milestone details table
        with st.expander("📋 View All Milestones Details"):
            display_df = milestones[['Milestone', 'Date', 'Status', 'Days_Behind']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%b %d, %Y')
            display_df.columns = ['Milestone', 'Date', 'Status', 'Days Behind']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Schedule recovery explanation
        st.markdown("""
        **Key Schedule Recovery Actions (May 10):**
        - 💰 **$100K/day bonus** for early completion (vs Sept 30 deadline)
        - 👷 Multiple erection crews deployed
        - 🏗️ Minimum 2 cranes instead of 1
        - 🔧 Two sets of traveling concrete forms instead of 1
        """)
    
    with tab2:
        st.subheader("Construction Sequence Progress")
        
        # Construction sequence data
        sequences = pd.DataFrame({
            'Sequence': ['I', 'II', 'III', 'IV', 'V'],
            'Location': [
                'Liberty St → Cortlandt St',
                'Cortlandt St → Dey St', 
                'Vesey St → Barclay St',
                'Greenwich & Cortlandt (Station)',
                'Dey St → North of Fulton St'
            ],
            'Steel_Complete_May': [100, 95, 100, 0, 0],
            'Steel_Complete_June': [100, 100, 100, 100, 50],
            'Steel_Complete_July': [100, 100, 100, 100, 100],
            'Concrete_Complete_June': [100, 95, 90, 90, 0],
            'Concrete_Complete_July': [100, 100, 100, 100, 95],
            'Track_Complete_July': [100, 100, 100, 100, 70]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Progress by month
            progress_data = pd.DataFrame({
                'Sequence': ['I', 'II', 'III', 'IV', 'V'] * 3,
                'Month': ['May'] * 5 + ['June'] * 5 + ['July'] * 5,
                'Steel_Progress': sequences['Steel_Complete_May'].tolist() + 
                                  sequences['Steel_Complete_June'].tolist() + 
                                  sequences['Steel_Complete_July'].tolist()
            })
            
            fig_steel = px.bar(progress_data, x='Sequence', y='Steel_Progress', color='Month',
                              barmode='group', title='Steel Erection Progress by Sequence',
                              color_discrete_sequence=['#E63946', '#F4A261', '#2A9D8F'])
            fig_steel.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=50, b=80),
                yaxis_title='% Complete',
                xaxis_title='Construction Sequence',
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_steel)
        
        with col2:
            # Final status by work type
            work_types = pd.DataFrame({
                'Work_Type': ['Steel Erection', 'Concrete', 'Track Work', 'Signals'],
                'Completion': [100, 100, 100, 100]
            })
            
            fig_work = px.bar(work_types, x='Work_Type', y='Completion',
                             title='Final Work Completion Status (Sept 15)',
                             color='Work_Type',
                             color_discrete_sequence=['#2E86AB', '#E63946', '#2A9D8F', '#9B5DE5'])
            fig_work.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=50, b=40),
                yaxis_title='% Complete',
                xaxis_title='Work Type',
                showlegend=False
            )
            st.plotly_chart(fig_work)
        
        # Project stats
        st.markdown("### 📊 Project Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Track Laid", "5 Miles")
        with stat_col2:
            st.metric("Heavy Cable", "45 Miles")
        with stat_col3:
            st.metric("Concrete Poured", "6,200 cu. yards")
        with stat_col4:
            st.metric("Steel Replaced", "1,050 Tons")
    
    with tab3:
        st.subheader("Project Float & Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Project Float over time (from Exhibit 9)
            float_data = pd.DataFrame({
                'Month': ['Feb-02', 'Mar-02', 'Apr-02', 'May-02', 'Jun-02', 'Jul-02', 'Aug-02', 'Sep-02'],
                'Float_Days': [30, 30, 15, 9, 13, 17, 22, 27]
            })
            
            fig_float = px.line(float_data, x='Month', y='Float_Days',
                               title='Project Float (Buffer to Sept 30 Deadline)',
                               markers=True)
            fig_float.update_traces(line_color='#2E86AB', marker_size=10)
            fig_float.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=50, b=40),
                yaxis_title='Float (Days)',
                xaxis_title='Month'
            )
            fig_float.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_float)
        
        with col2:
            # Work Complete vs Billed (from Exhibit 10)
            work_data = pd.DataFrame({
                'Month': ['Feb-02', 'Mar-02', 'Apr-02', 'May-02', 'Jun-02', 'Jul-02', 'Aug-02', 'Sep-02'],
                'Work_Complete': [0, 20, 40, 55, 70, 80, 90, 100],
                'Work_Billed': [0, 15, 35, 50, 65, 75, 85, 100]
            })
            
            fig_work = go.Figure()
            fig_work.add_trace(go.Scatter(x=work_data['Month'], y=work_data['Work_Complete'],
                                          mode='lines+markers', name='Work Complete',
                                          line=dict(color='#2A9D8F', width=3)))
            fig_work.add_trace(go.Scatter(x=work_data['Month'], y=work_data['Work_Billed'],
                                          mode='lines+markers', name='Work Billed',
                                          line=dict(color='#E63946', width=3)))
            fig_work.update_layout(
                title='Work Completed vs Billed',
                template='plotly_white',
                margin=dict(l=40, r=40, t=50, b=80),
                yaxis_title='% Complete',
                xaxis_title='Month',
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_work)
        
        # Financial Analysis
        st.markdown("### 💰 Financial Impact")
        
        fin_col1, fin_col2, fin_col3 = st.columns(3)
        with fin_col1:
            st.metric("Base Contract", "$159.5M")
        with fin_col2:
            st.metric("Early Completion Bonus", "$3M", "+15 days early")
        with fin_col3:
            st.metric("Affected Riders", "1.3M", help="Daily riders benefiting")
    
    with tab4:
        st.subheader("Key Project Management Learnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Success Factors
            
            **1. Incentive Alignment**
            - $100K/day bonus structure motivated contractor
            - Clear deadline (Sept 30) with accelerated target (Sept 1)
            - Win-win: MTA gets faster service, contractor earns bonus
            
            **2. Resource Commitment**
            - Multiple erection crews instead of one
            - Doubled crane capacity (2 vs 1)
            - Two concrete traveling forms
            - 24/7 work schedule (350 workers)
            
            **3. Parallel Work Streams**
            - Track/signals installed in completed sections
            - While construction continued on remaining phases
            - Testing began before all construction complete
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Unconventional Approaches
            
            **"Breaking the Rules"** (per Joe Trainor):
            
            - 📝 **As-Built Plans**: Drawings submitted AFTER construction
            - ⚡ **Permit Timing**: Work began before DOT permits approved
            - 🏃 **Fast-tracking**: Approval granted after the fact
            
            **Emotional Commitment**:
            - Workers didn't complain despite 100°F temperatures
            - Strong motivation beyond financial incentives
            - Symbolic importance of rebuilding after 9/11
            
            **Results**:
            - Project delays reduced from avg 20 months → 3 months
            - Completion 15 days before contractual deadline
            - Service restored for 1-year anniversary
            """)
        
        # Timeline comparison
        st.markdown("### ⏱️ Timeline Comparison")
        timeline_comp = pd.DataFrame({
            'Scenario': ['Initial Estimate', 'Contractual Deadline', 'Accelerated Target', 'Actual Completion'],
            'Duration_Months': [24, 8, 7, 7.5],
            'Color': ['#E63946', '#F4A261', '#2A9D8F', '#00A676']
        })
        
        fig_compare = px.bar(timeline_comp, x='Scenario', y='Duration_Months',
                             title='Project Duration Comparison',
                             color='Scenario',
                             color_discrete_sequence=['#E63946', '#F4A261', '#2A9D8F', '#00A676'])
        fig_compare.update_layout(
            template='plotly_white',
            margin=dict(l=40, r=40, t=50, b=40),
            yaxis_title='Duration (Months)',
            xaxis_title='',
            showlegend=False
        )
        st.plotly_chart(fig_compare)
    
    # Footer
    st.divider()
    st.caption("Source: London Business School Case Study 603-009-1 (April 2003)")
    
    print("DASHBOARD_READY", file=sys.stderr)
    
except Exception as e:
    print(f"DASHBOARD_ERROR: {str(e)}", file=sys.stderr)
    st.error(f"Error: {str(e)}")
