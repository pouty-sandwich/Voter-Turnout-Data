import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from datetime import datetime
import io
import json
import re
from anthropic import Anthropic
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize API clients safely - handle missing API keys
anthropic_client = None
openai_client = None

# Try to initialize Anthropic client
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
    else:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables")
except Exception as e:
    print(f"Failed to initialize Anthropic client: {e}")

# Try to initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")

# Simple Authentication System
def check_login():
    """Handle login screen and authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Show login form
        st.title("🔐 Voter Turnout Analyzer - Login")
        st.markdown("Please enter your credentials to access the voter analysis dashboard.")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                # Check credentials with your original password
                if username == "Votertrends" and password == 'ygG">pIA"95)wZ3':
                    st.session_state.authenticated = True
                    st.session_state['authentication_status'] = True
                    st.session_state['name'] = "Voter Trends User"
                    st.session_state['username'] = username
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
                    st.session_state['authentication_status'] = False
        
        return False  # Not authenticated
    
    else:
        # User is authenticated
        st.session_state['authentication_status'] = True
        return True  # Authenticated

# Session State Initialization
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}

# Helper Functions
@st.cache_data
def load_csv_efficiently(uploaded_file):
    try:
        chunk_list = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
            chunk_list.append(chunk)
        
        df = pd.concat(chunk_list, ignore_index=True)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def clean_numeric_column(series):
    def clean_value(val):
        if pd.isna(val):
            return 0
        val_str = str(val)
        cleaned_str = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(cleaned_str) if cleaned_str else 0
        except:
            return 0
    
    return series.apply(clean_value)

def find_column_by_keywords(df, keywords_list, priority_order=True):
    """Flexible column finder that matches based on keywords"""
    def normalize_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r'[^\w\s]', ' ', str(text).lower().strip())
    
    def calculate_match_score(column_name, keywords):
        normalized_col = normalize_text(column_name)
        score = 0
        
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword in normalized_col:
                if f" {normalized_keyword} " in f" {normalized_col} ":
                    score += 10
                else:
                    score += 5
        
        return score
    
    best_match = None
    best_score = 0
    
    for column in df.columns:
        for keywords in keywords_list:
            score = calculate_match_score(column, keywords)
            if score > best_score:
                best_score = score
                best_match = column
            
            if priority_order and score >= 10:
                return column
    
    return best_match if best_score > 0 else None

def detect_columns(df):
    """Intelligently detect column names with flexible matching"""
    try:
        column_info = {}
        
        # Precinct name detection
        precinct_keywords = [
            ['precinct', 'name'],
            ['precinct'],
            ['district', 'name'],
            ['ward', 'name'],
            ['location', 'name'],
            ['polling', 'place'],
            ['voting', 'location']
        ]
        column_info['precinct'] = find_column_by_keywords(df, precinct_keywords)
        
        # Registration total detection
        registration_keywords = [
            ['registration', 'total'],
            ['registered', 'total'],
            ['reg', 'total'],
            ['total', 'registration'],
            ['total', 'registered'],
            ['total', 'reg']
        ]
        column_info['registration_total'] = find_column_by_keywords(df, registration_keywords)
        
        # Vote count total detection
        vote_count_keywords = [
            ['public', 'count', 'total'],
            ['ballot', 'count', 'total'],
            ['vote', 'count', 'total'],
            ['votes', 'cast', 'total'],
            ['total', 'votes', 'cast'],
            ['total', 'ballots'],
            ['ballots', 'total'],
            ['turnout', 'total'],
            ['voted', 'total'],
            ['total', 'voted']
        ]
        
        vote_total_col = None
        for col in df.columns:
            col_lower = col.lower()
            if ('count' in col_lower or 'cast' in col_lower) and 'total' in col_lower and 'method' not in col_lower:
                vote_total_col = col
                break
        
        if not vote_total_col:
            vote_total_col = find_column_by_keywords(df, vote_count_keywords)
        
        column_info['vote_total'] = vote_total_col
        
        # Party registration and vote columns
        parties = ['dem', 'rep', 'republican', 'democrat', 'non', 'unaffiliated', 'independent']
        column_info['party_registration'] = {}
        column_info['party_votes'] = {}
        
        for party in parties:
            try:
                # Party registration
                party_reg_keywords = [
                    ['registration', party],
                    ['registered', party],
                    ['reg', party],
                    [party, 'registration'],
                    [party, 'registered'],
                    [party, 'reg']
                ]
                reg_col = find_column_by_keywords(df, party_reg_keywords)
                if reg_col:
                    party_standard = 'Dem' if party.lower() in ['dem', 'democrat'] else \
                                   'Rep' if party.lower() in ['rep', 'republican'] else \
                                   'Non' if party.lower() in ['non', 'unaffiliated', 'independent'] else party.title()
                    column_info['party_registration'][party_standard] = reg_col
                
                # Party vote counts
                party_vote_keywords = [
                    ['public', 'count', party],
                    ['vote', 'count', party],
                    ['votes', party],
                    [party, 'votes'],
                    [party, 'count'],
                    ['ballot', party],
                    [party, 'voted']
                ]
                vote_col = find_column_by_keywords(df, party_vote_keywords)
                if vote_col:
                    party_standard = 'Dem' if party.lower() in ['dem', 'democrat'] else \
                                   'Rep' if party.lower() in ['rep', 'republican'] else \
                                   'Non' if party.lower() in ['non', 'unaffiliated', 'independent'] else party.title()
                    column_info['party_votes'][party_standard] = vote_col
            except Exception:
                continue
        
        return column_info
    
    except Exception:
        return {
            'precinct': None,
            'registration_total': None,
            'vote_total': None,
            'party_registration': {},
            'party_votes': {}
        }

def analyze_dataset(df, dataset_name):
    debug_info = []
    debug_info.append(f"Total rows in file: {len(df)}")
    debug_info.append(f"Available columns: {list(df.columns)}")
    
    try:
        cols = detect_columns(df)
        debug_info.append("Column detection completed successfully")
        
        debug_info.append("Raw detection results:")
        debug_info.append(f"  - precinct: {cols.get('precinct')}")
        debug_info.append(f"  - registration_total: {cols.get('registration_total')}")
        debug_info.append(f"  - vote_total: {cols.get('vote_total')}")
        
    except Exception as e:
        debug_info.append(f"Column detection error: {e}")
        cols = {}
    
    precinct_col = cols.get('precinct')
    
    if not precinct_col:
        st.error("Could not find precinct column. Available columns: " + ", ".join(df.columns[:10]))
        return None
    
    # Get registration and vote columns with fallbacks
    reg_col = cols.get('registration_total')
    vote_col = cols.get('vote_total')
    
    if not reg_col:
        possible_reg_cols = [col for col in df.columns 
                           if any(word in col.lower() for word in ['registration', 'registered', 'reg']) 
                           and any(word in col.lower() for word in ['total', 'sum', 'all'])
                           and 'method' not in col.lower()]
        if possible_reg_cols:
            reg_col = possible_reg_cols[0]
    
    if not vote_col:
        possible_vote_cols = [col for col in df.columns 
                            if any(word in col.lower() for word in ['count', 'votes', 'voted', 'turnout']) 
                            and any(word in col.lower() for word in ['total', 'sum', 'all'])
                            and 'method' not in col.lower()]
        if possible_vote_cols:
            vote_col = possible_vote_cols[0]
    
    if not reg_col or not vote_col:
        st.error("Could not find required registration and vote columns")
        return None
    
    # Data aggregation strategy
    try:
        agg_dict = {}
        
        reg_columns = [col for col in df.columns if any(word in col.lower() for word in ['registration', 'registered', 'reg'])]
        for col in reg_columns:
            if col in df.columns:
                agg_dict[col] = 'max'
        
        count_columns = [col for col in df.columns if any(word in col.lower() for word in ['count', 'votes', 'voted', 'turnout'])]
        for col in count_columns:
            if col in df.columns:
                agg_dict[col] = 'sum'
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in agg_dict and col in df.columns:
                if any(word in col.lower() for word in ['registration', 'registered', 'reg']):
                    agg_dict[col] = 'max'
                else:
                    agg_dict[col] = 'sum'
        
        if len(agg_dict) > 0:
            df_aggregated = df.groupby(precinct_col).agg(agg_dict).reset_index()
            debug_info.append(f"Reduced from {len(df)} rows to {len(df_aggregated)} unique precincts")
            df = df_aggregated
    
    except Exception as e:
        debug_info.append(f"Aggregation warning: {e}")
    
    # Filter out summary rows
    summary_indicators = ['total', 'sum', 'grand', 'summary', 'citywide', 'combined', 'all precincts']
    
    filtered_df = df.copy()
    rows_removed = 0
    
    try:
        for indicator in summary_indicators:
            mask = filtered_df[precinct_col].astype(str).str.lower().str.contains(indicator, na=False)
            rows_with_indicator = mask.sum()
            if rows_with_indicator > 0:
                filtered_df = filtered_df[~mask]
                rows_removed += rows_with_indicator
    except Exception as e:
        debug_info.append(f"Summary filtering warning: {e}")
    
    try:
        # Clean and process the data
        reg_cleaned = clean_numeric_column(filtered_df[reg_col])
        vote_cleaned = clean_numeric_column(filtered_df[vote_col])
        
        total_registered = int(reg_cleaned.sum())
        total_voted = int(vote_cleaned.sum())
        total_rows = len(filtered_df)
        
        if total_registered == 0:
            st.error("No registration data found after processing")
            return None
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None
    
    # Party breakdown analysis
    party_stats = {}
    
    try:
        for party, reg_col_party in cols.get('party_registration', {}).items():
            vote_col_party = cols.get('party_votes', {}).get(party)
            
            if (reg_col_party and vote_col_party and 
                reg_col_party in filtered_df.columns and vote_col_party in filtered_df.columns):
                try:
                    party_reg_cleaned = clean_numeric_column(filtered_df[reg_col_party])
                    party_vote_cleaned = clean_numeric_column(filtered_df[vote_col_party])
                    
                    party_stats[party] = {
                        'registered': int(party_reg_cleaned.sum()),
                        'voted': int(party_vote_cleaned.sum())
                    }
                except Exception:
                    pass
    except Exception:
        pass
    
    # Create stats dictionary
    stats = {
        'name': dataset_name,
        'total_rows': total_rows,
        'total_registered': total_registered,
        'total_voted': total_voted,
        'registered_not_voted': max(0, total_registered - total_voted),
        'party_breakdown': party_stats,
        'reg_column_used': reg_col,
        'vote_column_used': vote_col,
        'rows_filtered': rows_removed,
        'debug_info': debug_info
    }
    
    stats['turnout_rate'] = (total_voted / total_registered * 100) if total_registered > 0 else 0
    
    return stats

def create_single_dataset_charts(stats):
    """Create comprehensive charts for a single dataset"""
    
    dataset_key = stats['name'].replace(' ', '_').replace('.', '_')
    
    # Row 1: Overview Charts
    st.subheader("📊 Overview Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Turnout breakdown pie chart
        fig_pie = px.pie(
            values=[stats['total_voted'], stats['registered_not_voted']],
            names=['Voted', 'Registered but Did Not Vote'],
            title=f"Voter Turnout Breakdown - {stats['name']}",
            color_discrete_sequence=['#2E8B57', '#FFD700']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{dataset_key}")
    
    with col2:
        # Bar chart of key metrics
        metrics = ['Total Precincts', 'Total Registered', 'Total Voted']
        values = [stats['total_rows'], stats['total_registered'], stats['total_voted']]
        
        fig_bar = px.bar(
            x=metrics, 
            y=values,
            title=f"Key Metrics - {stats['name']}",
            color=values,
            color_continuous_scale='viridis',
            text=values
        )
        fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{dataset_key}")
    
    # Row 2: Engagement Metrics
    st.subheader("🎯 Engagement Metrics")
    col3, col4 = st.columns(2)
    
    with col3:
        # Gauge chart for turnout rate
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = stats['turnout_rate'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Turnout Rate (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 70], 'color': "lightgreen"},
                    {'range': [70, 100], 'color': "darkgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_chart_{dataset_key}")
    
    with col4:
        # Donut chart for registration efficiency
        estimated_eligible = int(stats['total_registered'] / 0.7)
        reg_efficiency = (stats['total_registered'] / estimated_eligible) * 100
        non_registered = max(0, 100 - reg_efficiency)
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Registered', 'Potentially Unregistered'],
            values=[reg_efficiency, non_registered],
            hole=.5,
            marker_colors=['#1f77b4', '#d62728']
        )])
        fig_donut.update_layout(
            title="Registration Coverage Estimate",
            annotations=[dict(text=f'{reg_efficiency:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, key=f"donut_chart_{dataset_key}")
    
    # Party Analysis (if available)
    if stats['party_breakdown']:
        st.subheader("🎭 Party Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            parties = list(stats['party_breakdown'].keys())
            reg_values = [stats['party_breakdown'][party]['registered'] for party in parties]
            
            fig_party_reg = px.bar(
                x=parties,
                y=reg_values,
                title="Registration by Party",
                color=reg_values,
                color_continuous_scale='Blues',
                labels={'x': 'Party', 'y': 'Registered Voters'},
                text=reg_values
            )
            fig_party_reg.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_party_reg, use_container_width=True, key=f"party_reg_{dataset_key}")
        
        with col6:
            vote_values = [stats['party_breakdown'][party]['voted'] for party in parties]
            
            fig_party_vote = px.bar(
                x=parties,
                y=vote_values,
                title="Turnout by Party",
                color=vote_values,
                color_continuous_scale='Reds',
                labels={'x': 'Party', 'y': 'Votes Cast'},
                text=vote_values
            )
            fig_party_vote.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_party_vote, use_container_width=True, key=f"party_vote_{dataset_key}")

def create_comparison_charts(datasets_stats):
    if len(datasets_stats) < 2:
        return
    
    st.subheader("📊 Dataset Comparison")
    
    names = [stats['name'] for stats in datasets_stats]
    turnout_rates = [stats['turnout_rate'] for stats in datasets_stats]
    total_rows = [stats['total_rows'] for stats in datasets_stats]
    total_registered = [stats['total_registered'] for stats in datasets_stats]
    total_voted = [stats['total_voted'] for stats in datasets_stats]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_comp1 = px.bar(
            x=names, 
            y=turnout_rates,
            title="Turnout Rate Comparison",
            labels={'x': 'Dataset', 'y': 'Turnout Rate (%)'},
            color=turnout_rates,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_comp1, use_container_width=True, key="comparison_turnout")
    
    with col2:
        fig_comp2 = px.bar(
            x=names, 
            y=total_rows,
            title="Total Precincts Comparison",
            labels={'x': 'Dataset', 'y': 'Number of Precincts'},
            color=total_rows,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_comp2, use_container_width=True, key="comparison_precincts")

def create_export_section(datasets_stats):
    """Create export options section"""
    if not datasets_stats:
        return
    
    st.subheader("📄 Export Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export to JSON
        if st.button("📋 Export to JSON", help="Export data as JSON file"):
            export_data = []
            for stats in datasets_stats:
                export_data.append({
                    'dataset_name': stats['name'],
                    'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'summary': {
                        'total_precincts': stats['total_rows'],
                        'total_registered': stats['total_registered'],
                        'total_voted': stats['total_voted'],
                        'turnout_rate': round(stats['turnout_rate'], 2),
                        'registered_not_voted': stats['registered_not_voted']
                    },
                    'party_breakdown': stats.get('party_breakdown', {})
                })
            
            json_data = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="💾 Download JSON Report",
                data=json_data,
                file_name=f"voter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        # Export to CSV
        if st.button("📊 Export to CSV", help="Export summary data as CSV"):
            summary_data = []
            for stats in datasets_stats:
                row = {
                    'Dataset': stats['name'],
                    'Total_Precincts': stats['total_rows'],
                    'Total_Registered': stats['total_registered'],
                    'Total_Voted': stats['total_voted'],
                    'Turnout_Rate': round(stats['turnout_rate'], 2),
                    'Registration_Column': stats['reg_column_used'],
                    'Voting_Column': stats['vote_column_used']
                }
                
                if stats['party_breakdown']:
                    for party, data in stats['party_breakdown'].items():
                        row[f'{party}_Registered'] = data['registered']
                        row[f'{party}_Voted'] = data['voted']
                        row[f'{party}_Turnout_Rate'] = round((data['voted'] / data['registered'] * 100) if data['registered'] > 0 else 0, 2)
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            csv_data = summary_df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV Summary",
                data=csv_data,
                file_name=f"voter_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Login Interface
if check_login():
    pass
else:
    st.stop()

# Main Application
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state['authentication_status'] = False
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.rerun()

st.title("🗳️ Voter Turnout Analyzer")
st.markdown(f"**Welcome, {st.session_state['name']}!** Upload and analyze voter turnout data with advanced visualizations.")

with st.sidebar:
    st.header("📁 Dataset Management")
    
    if st.session_state['datasets']:
        st.subheader("Loaded Datasets:")
        for name in st.session_state['datasets'].keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {name}")
            with col2:
                if st.button("🗑️", key=f"delete_{name}", help=f"Delete {name}"):
                    del st.session_state['datasets'][name]
                    st.rerun()
    
    if st.button("Clear All Datasets"):
        st.session_state['datasets'] = {}
        st.rerun()

st.subheader("📤 Upload Voter CSV Files")
uploaded_files = st.file_uploader(
    "Choose CSV files", 
    type="csv", 
    accept_multiple_files=True,
    help="Upload one or more CSV files with election data - columns will be automatically detected"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        dataset_name = uploaded_file.name.replace('.csv', '')
        
        if dataset_name not in st.session_state['datasets']:
            with st.spinner(f"Loading {dataset_name}..."):
                df = load_csv_efficiently(uploaded_file)
                
                if df is not None:
                    stats = analyze_dataset(df, dataset_name)
                    if stats:
                        st.session_state['datasets'][dataset_name] = {
                            'data': df,
                            'stats': stats
                        }
                        st.success(f"✅ Loaded {dataset_name} ({len(df):,} records)")
                    else:
                        st.error(f"❌ {dataset_name} could not be processed")

if st.session_state['datasets']:
    st.subheader("📊 Data Analysis & Visualizations")
    
    for dataset_name, dataset_info in st.session_state['datasets'].items():
        with st.expander(f"📈 Analysis: {dataset_name}", expanded=len(st.session_state['datasets']) == 1):
            stats = dataset_info['stats']
            df = dataset_info['data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Precincts", f"{stats['total_rows']:,}")
            with col2:
                st.metric("Total Registered", f"{stats['total_registered']:,}")
            with col3:
                st.metric("Total Voted", f"{stats['total_voted']:,}")
            with col4:
                st.metric("Turnout Rate", f"{stats['turnout_rate']:.2f}%")
            
            st.info(f"📊 **Data Source**: Registration from '{stats['reg_column_used']}', Voting from '{stats['vote_column_used']}'")
            
            if stats.get('rows_filtered', 0) > 0:
                st.info(f"🔧 **Data Cleaning**: Filtered out {stats['rows_filtered']} summary/outlier rows for accurate counting")
            
            if 'debug_info' in stats:
                with st.expander("🔍 Technical Details & Data Processing", expanded=False):
                    st.write("**Data Processing Steps:**")
                    for info in stats['debug_info']:
                        st.write(f"- {info}")
            
            if stats['party_breakdown']:
                st.write("**Party Breakdown:**")
                party_cols = st.columns(len(stats['party_breakdown']))
                for i, (party, data) in enumerate(stats['party_breakdown'].items()):
                    with party_cols[i]:
                        turnout_rate = (data['voted'] / data['registered'] * 100) if data['registered'] > 0 else 0
                        st.metric(
                            f"{party} Turnout", 
                            f"{turnout_rate:.1f}%",
                            f"{data['voted']:,} of {data['registered']:,}"
                        )
            
            st.write("**Data Preview:**")
            st.dataframe(df.head(100), use_container_width=True)
            
            create_single_dataset_charts(stats)
            
            # AI SUGGESTION BUTTON - CORRECTLY PLACED HERE
            if st.button(f"🤖 Get AI Suggestions for {dataset_name}", key=f"ai_{dataset_name}"):
                prompt = (
                    f"As an expert in election data synthesis and civic engagement, analyze this election data from {dataset_name}:\n\n"
                    f"Total Precincts: {stats['total_rows']:,}\n"
                    f"Total Registered: {stats['total_registered']:,}\n"
                    f"Total Voted: {stats['total_voted']:,}\n"
                    f"Overall Turnout Rate: {stats['turnout_rate']:.2f}%\n"
                    + (f"\nParty Breakdown:\n" + "\n".join([
                        f"- {party}: {data['voted']:,} voted out of {data['registered']:,} registered ({data['voted']/data['registered']*100:.1f}% turnout)"
                        for party, data in stats['party_breakdown'].items() if data['registered'] > 0
                    ]) if stats['party_breakdown'] else "") +
                    f"\n\nPlease provide:\n"
                    f"1. What are 3-4 other large cities that historically struggled with voter turnout similar to this rate ({stats['turnout_rate']:.1f}%) but then significantly increased their turnout in subsequent elections?\n"
                    f"2. What specific, concrete steps did those cities take to increase voter participation?\n"
                    f"3. Which of those strategies would be most applicable to this jurisdiction based on the data patterns shown?\n"
                    f"\nFocus on real examples with measurable results and specific implementation strategies."
                )
                
                success = False
                
                # Try Anthropic first if available
                if anthropic_client:
                    try:
                        response = anthropic_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=1000,
                            messages=[
                                {"role": "user", "content": f"You are a civic engagement expert specializing in voter turnout analysis.\n\n{prompt}"}
                            ]
                        )
                        
                        if response:
                            suggestions = response.content[0].text
                            st.markdown("### 🤖 AI-Generated Improvement Suggestions (Claude)")
                            st.write(suggestions)
                            success = True
                            
                    except Exception as anthropic_error:
                        st.warning(f"Anthropic API failed: {anthropic_error}")
                
                # Try OpenAI if Anthropic failed or wasn't available
                if not success and openai_client:
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            max_tokens=1000,
                            messages=[
                                {"role": "system", "content": "You are a civic engagement expert specializing in voter turnout analysis."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        if response:
                            suggestions = response.choices[0].message.content
                            st.markdown("### 🤖 AI-Generated Improvement Suggestions (GPT)")
                            st.write(suggestions)
                            success = True
                            
                    except Exception as openai_error:
                        st.error(f"OpenAI API failed: {openai_error}")
                
                # If no AI service is available
                if not success:
                    if not anthropic_client and not openai_client:
                        st.error("🚫 No AI services available. Please check your API keys in the environment variables.")
                        st.info("Required environment variables: ANTHROPIC_API_KEY and/or OPENAI_API_KEY")
                    else:
                        st.error("🚫 All available AI services failed. Please check your API keys and try again.")
    
    if len(st.session_state['datasets']) > 1:
        datasets_stats = [info['stats'] for info in st.session_state['datasets'].values()]
        create_comparison_charts(datasets_stats)
    
    # Add export functionality
    datasets_stats = [info['stats'] for info in st.session_state['datasets'].values()]
    create_export_section(datasets_stats)

else:
    st.info("👆 Upload CSV files to begin analysis")

# Authentication status messages
if st.session_state['authentication_status'] is False:
    st.error("❌ Username/password is incorrect")
elif st.session_state['authentication_status'] is None:
    st.warning("🔐 Please enter your username and password")
