import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure Streamlit
st.set_page_config(
    page_title="Voter Turnout Analyzer", 
    page_icon="🗳️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Authentication (no external dependencies)
def check_password():
    def password_entered():
        if st.session_state["password"] == "voter2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😞 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Set up simple session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

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

def analyze_precinct_performance(df, precinct_col, reg_col, vote_col):
    """Analyze individual precinct performance"""
    try:
        precinct_analysis = df.groupby(precinct_col).agg({
            reg_col: 'first',  # Registration is max per precinct
            vote_col: 'first'  # Already aggregated votes
        }).reset_index()
        
        # Clean the data
        precinct_analysis[reg_col] = clean_numeric_column(precinct_analysis[reg_col])
        precinct_analysis[vote_col] = clean_numeric_column(precinct_analysis[vote_col])
        
        # Calculate turnout rate
        precinct_analysis['turnout_rate'] = (
            precinct_analysis[vote_col] / precinct_analysis[reg_col] * 100
        ).fillna(0)
        
        # Performance tiers
        precinct_analysis['performance_tier'] = pd.cut(
            precinct_analysis['turnout_rate'], 
            bins=[0, 40, 60, 80, 100], 
            labels=['Needs Attention', 'Below Average', 'Good', 'Excellent']
        )
        
        return precinct_analysis.sort_values('turnout_rate', ascending=False)
    except Exception as e:
        st.warning(f"Could not analyze precinct performance: {e}")
        return None

def analyze_voting_methods(df, vote_method_col, reg_col, vote_col):
    """Analyze turnout by voting method"""
    try:
        method_stats = {}
        
        for method in df[vote_method_col].unique():
            if pd.isna(method):
                continue
                
            method_data = df[df[vote_method_col] == method]
            total_reg = clean_numeric_column(method_data[reg_col]).sum()
            total_votes = clean_numeric_column(method_data[vote_col]).sum()
            
            method_stats[str(method)] = {
                'precincts': len(method_data),
                'total_registered': int(total_reg),
                'total_voted': int(total_votes),
                'avg_turnout_rate': (total_votes / total_reg * 100) if total_reg > 0 else 0
            }
        
        return method_stats
    except Exception as e:
        st.warning(f"Could not analyze voting methods: {e}")
        return {}

def identify_turnout_hotspots(precinct_analysis):
    """Identify high and low performance clusters"""
    if precinct_analysis is None or len(precinct_analysis) == 0:
        return {}
    
    try:
        # Top and bottom performers
        top_count = max(1, int(len(precinct_analysis) * 0.1))
        bottom_count = max(1, int(len(precinct_analysis) * 0.1))
        
        top_performers = precinct_analysis.head(top_count)
        bottom_performers = precinct_analysis.tail(bottom_count)
        
        hotspots = {
            'high_performers': {
                'count': len(top_performers),
                'precincts': top_performers.iloc[:, 0].tolist()[:5],  # First 5 names
                'avg_turnout': top_performers['turnout_rate'].mean(),
                'min_turnout': top_performers['turnout_rate'].min(),
                'max_turnout': top_performers['turnout_rate'].max()
            },
            'low_performers': {
                'count': len(bottom_performers),
                'precincts': bottom_performers.iloc[:, 0].tolist()[:5],  # First 5 names
                'avg_turnout': bottom_performers['turnout_rate'].mean(),
                'min_turnout': bottom_performers['turnout_rate'].min(),
                'max_turnout': bottom_performers['turnout_rate'].max()
            }
        }
        
        return hotspots
    except Exception as e:
        st.warning(f"Could not identify hotspots: {e}")
        return {}

def analyze_registration_efficiency(stats):
    """Analyze registration and turnout efficiency"""
    try:
        # More realistic eligible population estimate
        # Option 1: Conservative estimate based on registration data (assuming 60-80% registration rate)
        estimated_eligible_conservative = int(stats['total_registered'] / 0.7)  # Assume 70% registration rate
        
        # Option 2: Alternative based on typical voting-age population ratios (75% of total population)
        # For comparison, but we use the conservative estimate
        
        efficiency_metrics = {
            'estimated_eligible': estimated_eligible_conservative,
            'registration_rate': (stats['total_registered'] / estimated_eligible_conservative) * 100,
            'voting_rate_of_eligible': (stats['total_voted'] / estimated_eligible_conservative) * 100,
            'voting_rate_of_registered': stats['turnout_rate'],
            'registration_gap': estimated_eligible_conservative - stats['total_registered'],
            'participation_gap': stats['total_registered'] - stats['total_voted'],
            'potential_new_voters': max(0, estimated_eligible_conservative - stats['total_registered']),
            'potential_turnout_improvement': max(0, stats['total_registered'] - stats['total_voted'])
        }
        
        return efficiency_metrics
    except Exception as e:
        st.warning(f"Could not analyze registration efficiency: {e}")
        return {}

def benchmark_analysis(stats):
    """Compare to benchmarks"""
    try:
        # Realistic benchmarks based on typical US elections
        benchmarks = {
            'excellent_turnout': 80,
            'good_turnout': 65,
            'average_turnout': 50,
            'presidential_avg': 60,
            'midterm_avg': 45,
            'local_avg': 35
        }
        
        performance = {}
        for benchmark_name, benchmark_value in benchmarks.items():
            difference = stats['turnout_rate'] - benchmark_value
            performance[benchmark_name] = {
                'benchmark_value': benchmark_value,
                'difference': difference,
                'performance': 'Above' if difference > 0 else 'Below',
                'percentage_diff': (difference / benchmark_value * 100) if benchmark_value > 0 else 0
            }
        
        return performance
    except Exception as e:
        st.warning(f"Could not perform benchmark analysis: {e}")
        return {}

def find_column_by_keywords(df, keywords_list, priority_order=True):
    """
    Flexible column finder that matches based on keywords
    
    Args:
        df: DataFrame to search
        keywords_list: List of keyword sets to search for
        priority_order: If True, return first match; if False, return best match
    
    Returns:
        Column name if found, None if not found
    """
    import re
    
    def normalize_text(text):
        """Normalize text for comparison"""
        if pd.isna(text):
            return ""
        return re.sub(r'[^\w\s]', ' ', str(text).lower().strip())
    
    def calculate_match_score(column_name, keywords):
        """Calculate how well a column name matches keywords"""
        normalized_col = normalize_text(column_name)
        score = 0
        
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword in normalized_col:
                # Exact word match gets higher score
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
            
            # If priority_order is True and we found a good match, return it
            if priority_order and score >= 10:
                return column
    
    return best_match if best_score > 0 else None

def detect_columns(df):
    """
    Intelligently detect column names with flexible matching
    """
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
        
        # Vote method detection
        method_keywords = [
            ['vote', 'method'],
            ['voting', 'method'],
            ['method'],
            ['vote', 'type'],
            ['voting', 'type'],
            ['ballot', 'type']
        ]
        column_info['vote_method'] = find_column_by_keywords(df, method_keywords)
        
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
        
        # Vote count total detection - be more specific to avoid "Vote Method"
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
        # Specifically look for columns that contain "count" or "cast" but NOT "method"
        vote_total_col = None
        for col in df.columns:
            col_lower = col.lower()
            if ('count' in col_lower or 'cast' in col_lower) and 'total' in col_lower and 'method' not in col_lower:
                vote_total_col = col
                break
        
        if not vote_total_col:
            vote_total_col = find_column_by_keywords(df, vote_count_keywords)
        
        column_info['vote_total'] = vote_total_col
        
        # Party registration columns
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
                    # Standardize party name
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
            except Exception as e:
                # Continue if one party fails
                continue
        
        # Date of birth detection
        dob_keywords = [
            ['date', 'birth'],
            ['birth', 'date'],
            ['dob'],
            ['birthdate'],
            ['birth_date'],
            ['date_of_birth'],
            ['born'],
            ['birthday']
        ]
        column_info['date_of_birth'] = find_column_by_keywords(df, dob_keywords)
        
        return column_info
    
    except Exception as e:
        # Return empty dict if detection fails completely
        return {
            'precinct': None,
            'vote_method': None,
            'registration_total': None,
            'vote_total': None,
            'party_registration': {},
            'party_votes': {},
            'date_of_birth': None
        }

def calculate_age_from_dob(dob_series, reference_date=None):
    """
    Calculate age from date of birth
    
    Args:
        dob_series: Pandas series with date of birth values
        reference_date: Reference date for age calculation (default: today)
    
    Returns:
        Pandas series with calculated ages
    """
    from datetime import datetime
    import pandas as pd
    
    if reference_date is None:
        reference_date = datetime.now()
    
    def safe_age_calc(dob):
        try:
            if pd.isna(dob):
                return None
            
            # Convert to datetime if it's a string
            if isinstance(dob, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y']:
                    try:
                        dob = datetime.strptime(dob, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format worked, try pandas to_datetime
                    dob = pd.to_datetime(dob, errors='coerce')
            
            if pd.isna(dob):
                return None
                
            # Calculate age
            age = reference_date.year - dob.year
            if (reference_date.month, reference_date.day) < (dob.month, dob.day):
                age -= 1
                
            return age if 0 <= age <= 150 else None  # Sanity check
            
        except Exception:
            return None
    
    return dob_series.apply(safe_age_calc)

def categorize_age_groups(age_series):
    """
    Categorize ages into generational groups
    
    Args:
        age_series: Pandas series with ages
    
    Returns:
        Pandas series with age group categories
    """
    def get_age_group(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 18:
            return 'Under 18'
        elif 18 <= age <= 26:
            return 'Gen Z (18-26)'
        elif 27 <= age <= 42:
            return 'Millennial (27-42)'
        elif 43 <= age <= 58:
            return 'Gen X (43-58)'
        elif 59 <= age <= 77:
            return 'Boomer (59-77)'
        elif age >= 78:
            return 'Silent (78+)'
        else:
            return 'Unknown'
    
    return age_series.apply(get_age_group)

def analyze_age_demographics(df, dob_col, reg_col, vote_col):
    """
    Analyze voter turnout by age demographics
    
    Args:
        df: DataFrame with voter data
        dob_col: Date of birth column name
        reg_col: Registration column name
        vote_col: Vote count column name
    
    Returns:
        Dictionary with age-based analysis
    """
    try:
        # Calculate ages
        ages = calculate_age_from_dob(df[dob_col])
        age_groups = categorize_age_groups(ages)
        
        # Create temporary dataframe for analysis
        temp_df = df.copy()
        temp_df['calculated_age'] = ages
        temp_df['age_group'] = age_groups
        
        # Remove rows with unknown ages for analysis
        valid_age_df = temp_df[temp_df['age_group'] != 'Unknown'].copy()
        
        if len(valid_age_df) == 0:
            return {}
        
        # Group by age group and sum registration/votes
        age_stats = valid_age_df.groupby('age_group').agg({
            reg_col: 'sum',
            vote_col: 'sum'
        }).reset_index()
        
        # Calculate turnout rates
        age_stats['turnout_rate'] = (age_stats[vote_col] / age_stats[reg_col] * 100).fillna(0)
        
        # Convert to dictionary format
        age_analysis = {}
        for _, row in age_stats.iterrows():
            age_group = row['age_group']
            age_analysis[age_group] = {
                'registered': int(row[reg_col]),
                'voted': int(row[vote_col]),
                'turnout_rate': float(row['turnout_rate'])
            }
        
        # Add summary statistics
        total_with_known_age = valid_age_df[reg_col].sum()
        total_voted_with_known_age = valid_age_df[vote_col].sum()
        
        age_analysis['_summary'] = {
            'total_with_known_age': int(total_with_known_age),
            'total_voted_with_known_age': int(total_voted_with_known_age),
            'coverage_percentage': (len(valid_age_df) / len(df) * 100) if len(df) > 0 else 0,
            'avg_age': float(ages.mean()) if not ages.isna().all() else None,
            'median_age': float(ages.median()) if not ages.isna().all() else None
        }
        
        return age_analysis
        
    except Exception as e:
        st.warning(f"Could not analyze age demographics: {e}")
        return {}

def analyze_dataset(df, dataset_name):
    # Use flexible column detection
    debug_info = []
    debug_info.append(f"Total rows in file: {len(df)}")
    debug_info.append(f"Available columns: {list(df.columns)}")  # Show all columns for debugging
    
    try:
        cols = detect_columns(df)
        debug_info.append(f"Column detection completed successfully")
        
        # Show what was actually detected
        debug_info.append(f"Raw detection results:")
        debug_info.append(f"  - precinct: {cols.get('precinct')}")
        debug_info.append(f"  - vote_method: {cols.get('vote_method')}")  
        debug_info.append(f"  - registration_total: {cols.get('registration_total')}")
        debug_info.append(f"  - vote_total: {cols.get('vote_total')}")
        debug_info.append(f"  - date_of_birth: {cols.get('date_of_birth')}")
        
    except Exception as e:
        debug_info.append(f"Column detection error: {e}")
        cols = {}
    
    vote_method_col = cols.get('vote_method')
    precinct_col = cols.get('precinct')
    
    # Report detected columns
    debug_info.append(f"Detected precinct column: {precinct_col}")
    debug_info.append(f"Detected vote method column: {vote_method_col}")
    debug_info.append(f"Detected registration total: {cols.get('registration_total')}")
    debug_info.append(f"Detected vote total: {cols.get('vote_total')}")
    debug_info.append(f"Detected date of birth column: {cols.get('date_of_birth')}")
    
    if not precinct_col:
        st.error("Could not find precinct column. Available columns: " + ", ".join(df.columns[:10]))
        debug_info.append("No precinct column found - analysis cannot continue")
        return None
    
    # Get registration and vote columns with fallbacks
    reg_col = cols.get('registration_total')
    vote_col = cols.get('vote_total')
    
    debug_info.append(f"Initial detection - reg_col: {reg_col}, vote_col: {vote_col}")
    
    if not reg_col:
        # Fallback: search more broadly for registration
        possible_reg_cols = [col for col in df.columns 
                           if any(word in col.lower() for word in ['registration', 'registered', 'reg']) 
                           and any(word in col.lower() for word in ['total', 'sum', 'all'])
                           and 'method' not in col.lower()]  # Exclude method columns
        if possible_reg_cols:
            reg_col = possible_reg_cols[0]
            debug_info.append(f"Fallback registration column found: {reg_col}")
    
    if not vote_col:
        # Fallback: search more broadly for vote counts (but NOT vote method)
        possible_vote_cols = [col for col in df.columns 
                            if any(word in col.lower() for word in ['count', 'votes', 'voted', 'turnout']) 
                            and any(word in col.lower() for word in ['total', 'sum', 'all'])
                            and 'method' not in col.lower()]  # Exclude method columns
        if possible_vote_cols:
            vote_col = possible_vote_cols[0]
            debug_info.append(f"Fallback vote column found: {vote_col}")
    
    debug_info.append(f"Final columns - reg_col: {reg_col}, vote_col: {vote_col}")
    
    if not reg_col:
        st.error("Could not find registration column. Available columns: " + ", ".join([col for col in df.columns if 'reg' in col.lower() or 'registration' in col.lower()]))
        return None
    
    if not vote_col:
        st.error("Could not find vote count column. Available columns: " + ", ".join([col for col in df.columns if 'count' in col.lower() or 'votes' in col.lower()]))
        return None
    
    # Report unique values for debugging
    unique_precincts = df[precinct_col].nunique()
    debug_info.append(f"Unique precincts: {unique_precincts}")
    
    if vote_method_col and vote_method_col in df.columns:
        try:
            # Safe access to vote method column
            vote_method_series = df[vote_method_col]
            unique_methods = vote_method_series.nunique()
            debug_info.append(f"Unique vote methods: {unique_methods}")
            
            # Get unique values safely
            unique_values = vote_method_series.dropna().unique()
            debug_info.append(f"Vote methods: {list(unique_values[:5])}")  # Show first 5
        except Exception as e:
            debug_info.append(f"Issue with vote method column '{vote_method_col}': {e}")
            vote_method_col = None  # Disable if problematic
    elif vote_method_col:
        debug_info.append(f"Vote method column '{vote_method_col}' not found in dataframe")
        vote_method_col = None
    
    # Data aggregation strategy
    try:
        agg_dict = {}
        
        # Handle registration columns (take max per precinct)
        reg_columns = [col for col in df.columns if any(word in col.lower() for word in ['registration', 'registered', 'reg'])]
        for col in reg_columns:
            if col in df.columns:
                agg_dict[col] = 'max'
        
        # Handle vote count columns (sum per precinct)
        count_columns = [col for col in df.columns if any(word in col.lower() for word in ['count', 'votes', 'voted', 'turnout'])]
        for col in count_columns:
            if col in df.columns:
                agg_dict[col] = 'sum'
        
        # Handle other numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in agg_dict and col in df.columns:
                if any(word in col.lower() for word in ['registration', 'registered', 'reg']):
                    agg_dict[col] = 'max'
                else:
                    agg_dict[col] = 'sum'
        
        debug_info.append("Aggregation strategy:")
        debug_info.append(f"MAX (registration): {[k for k,v in agg_dict.items() if v == 'max'][:5]}")
        debug_info.append(f"SUM (vote counts): {[k for k,v in agg_dict.items() if v == 'sum'][:5]}")
        
        # Perform aggregation by precinct
        if len(agg_dict) > 0:
            df_aggregated = df.groupby(precinct_col).agg(agg_dict).reset_index()
            debug_info.append(f"Reduced from {len(df)} rows to {len(df_aggregated)} unique precincts")
            df = df_aggregated
        else:
            debug_info.append("No aggregation needed - using original data")
    
    except Exception as e:
        debug_info.append(f"Aggregation warning: {e}")
        # Continue with original data if aggregation fails
    
    # Filter out summary rows
    summary_indicators = ['total', 'sum', 'grand', 'summary', 'citywide', 'combined', 'all precincts']
    
    filtered_df = df.copy()
    rows_removed = 0
    
    try:
        for indicator in summary_indicators:
            mask = filtered_df[precinct_col].astype(str).str.lower().str.contains(indicator, na=False)
            rows_with_indicator = mask.sum()
            if rows_with_indicator > 0:
                debug_info.append(f"Removing {rows_with_indicator} rows containing '{indicator}'")
                filtered_df = filtered_df[~mask]
                rows_removed += rows_with_indicator
    except Exception as e:
        debug_info.append(f"Summary filtering warning: {e}")
    
    # Use the detected/fallback columns
    total_reg_col = reg_col
    total_vote_col = vote_col
    
    # Safety check: make sure we never use vote method as vote total
    if total_vote_col == vote_method_col:
        debug_info.append(f"ERROR: Vote total column incorrectly set to vote method column. Searching for alternative.")
        # Find an alternative vote total column
        alt_vote_cols = [col for col in df.columns 
                        if ('count' in col.lower() or 'cast' in col.lower()) 
                        and 'total' in col.lower() 
                        and col != vote_method_col]
        if alt_vote_cols:
            total_vote_col = alt_vote_cols[0]
            debug_info.append(f"Found alternative vote column: {total_vote_col}")
        else:
            st.error("Could not find a valid vote count column that is different from vote method column")
            return None
    
    debug_info.append(f"Using columns: reg='{total_reg_col}', vote='{total_vote_col}', method='{vote_method_col}'")
    
    # Verify columns actually exist in the filtered dataframe
    if total_reg_col not in filtered_df.columns:
        st.error(f"Registration column '{total_reg_col}' not found in data after filtering")
        debug_info.append(f"Available columns after filtering: {list(filtered_df.columns)}")
        return None
    
    if total_vote_col not in filtered_df.columns:
        st.error(f"Vote column '{total_vote_col}' not found in data after filtering")
        debug_info.append(f"Available columns after filtering: {list(filtered_df.columns)}")
        return None
    
    try:
        # Clean and process the data
        reg_cleaned = clean_numeric_column(filtered_df[total_reg_col])
        vote_cleaned = clean_numeric_column(filtered_df[total_vote_col])
        
        total_registered = int(reg_cleaned.sum())
        total_voted = int(vote_cleaned.sum())
        total_rows = len(filtered_df)
        
        debug_info.append(f"Final Results: {total_rows} precincts, {total_registered:,} registered, {total_voted:,} voted")
        debug_info.append(f"Sample registration values: {[float(x) for x in reg_cleaned.head().tolist()]}")
        debug_info.append(f"Sample vote count values: {[float(x) for x in vote_cleaned.head().tolist()]}")
        
        if total_registered == 0:
            st.error("No registration data found after processing")
            return None
        
        if total_voted == 0:
            st.warning("No vote data found after processing")
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        debug_info.append(f"Processing error details: {str(e)}")
        
        # Safe debug information - check if columns exist first
        try:
            if total_reg_col and total_reg_col in filtered_df.columns:
                debug_info.append(f"Registration column '{total_reg_col}' sample: {filtered_df[total_reg_col].head().tolist()}")
            else:
                debug_info.append(f"Registration column '{total_reg_col}' not found in filtered data")
        except Exception as debug_e:
            debug_info.append(f"Could not show registration sample: {debug_e}")
        
        try:
            if total_vote_col and total_vote_col in filtered_df.columns:
                debug_info.append(f"Vote column '{total_vote_col}' sample: {filtered_df[total_vote_col].head().tolist()}")
            else:
                debug_info.append(f"Vote column '{total_vote_col}' not found in filtered data")
        except Exception as debug_e:
            debug_info.append(f"Could not show vote sample: {debug_e}")
        
        # Show available columns for debugging
        debug_info.append(f"Available columns in filtered data: {list(filtered_df.columns)}")
        
        return None
    
    # Party breakdown analysis using detected columns
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
                except Exception as e:
                    debug_info.append(f"Could not process {party} party data: {e}")
    except Exception as e:
        debug_info.append(f"Party analysis warning: {e}")
    
    # Create stats dictionary
    stats = {
        'name': dataset_name,
        'total_rows': total_rows,
        'total_registered': total_registered,
        'total_voted': total_voted,
        'registered_not_voted': max(0, total_registered - total_voted),
        'party_breakdown': party_stats,
        'reg_column_used': total_reg_col,
        'vote_column_used': total_vote_col,
        'rows_filtered': rows_removed,
        'debug_info': debug_info
    }
    
    stats['turnout_rate'] = (total_voted / total_registered * 100) if total_registered > 0 else 0
    
    # Enhanced Analysis - Add new analytics with better error handling
    try:
        # Precinct performance analysis
        if precinct_col and total_reg_col and total_vote_col:
            try:
                precinct_performance = analyze_precinct_performance(filtered_df, precinct_col, total_reg_col, total_vote_col)
                if precinct_performance is not None:
                    stats['precinct_performance'] = precinct_performance
                    
                    # Hotspot analysis
                    hotspots = identify_turnout_hotspots(precinct_performance)
                    stats['hotspots'] = hotspots
            except Exception as e:
                debug_info.append(f"Precinct analysis error: {e}")
        
        # Voting method analysis
        if vote_method_col and vote_method_col in filtered_df.columns:
            try:
                method_analysis = analyze_voting_methods(filtered_df, vote_method_col, total_reg_col, total_vote_col)
                if method_analysis:
                    stats['voting_methods'] = method_analysis
            except Exception as e:
                debug_info.append(f"Voting method analysis error: {e}")
        
        # Age demographics analysis
        dob_col = cols.get('date_of_birth')
        if dob_col and dob_col in filtered_df.columns:
            try:
                age_analysis = analyze_age_demographics(filtered_df, dob_col, total_reg_col, total_vote_col)
                if age_analysis:
                    stats['age_demographics'] = age_analysis
                    debug_info.append(f"Age analysis completed for {len(age_analysis)} age groups")
            except Exception as e:
                debug_info.append(f"Age analysis error: {e}")
        
        # Registration efficiency analysis
        try:
            efficiency_analysis = analyze_registration_efficiency(stats)
            if efficiency_analysis:
                stats['efficiency_metrics'] = efficiency_analysis
        except Exception as e:
            debug_info.append(f"Efficiency analysis error: {e}")
        
        # Benchmark analysis
        try:
            benchmark_results = benchmark_analysis(stats)
            if benchmark_results:
                stats['benchmarks'] = benchmark_results
        except Exception as e:
            debug_info.append(f"Benchmark analysis error: {e}")
        
    except Exception as e:
        debug_info.append(f"Enhanced analysis warning: {e}")
    
    return stats

def create_single_dataset_charts(stats):
    """Create comprehensive charts for a single dataset"""
    
    # Create a unique key prefix based on dataset name
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
        estimated_eligible_streamlit = int(stats['total_registered'] / 0.7)  # Assume 70% registration rate
        reg_efficiency = (stats['total_registered'] / estimated_eligible_streamlit) * 100
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
    
    # Row 3: Party Analysis (if available)
    if stats['party_breakdown']:
        st.subheader("🎭 Party Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            # Party registration comparison
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
            # Party turnout comparison
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
        
        # Row 4: Party Performance
        col7, col8 = st.columns(2)
        
        with col7:
            # Party turnout rates
            party_turnout_rates = []
            party_names = []
            for party, data in stats['party_breakdown'].items():
                if data['registered'] > 0:
                    rate = (data['voted'] / data['registered']) * 100
                    party_turnout_rates.append(rate)
                    party_names.append(party)
            
            if party_turnout_rates:
                fig_party_rates = px.bar(
                    x=party_names,
                    y=party_turnout_rates,
                    title="Turnout Rate by Party (%)",
                    color=party_turnout_rates,
                    color_continuous_scale='RdYlGn',
                    labels={'x': 'Party', 'y': 'Turnout Rate (%)'},
                    text=[f'{rate:.1f}%' for rate in party_turnout_rates]
                )
                fig_party_rates.update_traces(textposition='outside')
                st.plotly_chart(fig_party_rates, use_container_width=True, key=f"party_rates_{dataset_key}")
        
        with col8:
            # Stacked bar chart showing party composition
            fig_stacked = go.Figure()
            fig_stacked.add_trace(go.Bar(
                name='Registered',
                x=parties,
                y=reg_values,
                marker_color='lightblue',
                text=[f'{val:,.0f}' for val in reg_values],
                textposition='inside'
            ))
            fig_stacked.add_trace(go.Bar(
                name='Voted',
                x=parties,
                y=vote_values,
                marker_color='darkgreen',
                text=[f'{val:,.0f}' for val in vote_values],
                textposition='inside'
            ))
            
            fig_stacked.update_layout(
                title='Registration vs Voting by Party',
                barmode='group',
                xaxis_title='Party',
                yaxis_title='Count'
            )
            st.plotly_chart(fig_stacked, use_container_width=True, key=f"party_stacked_{dataset_key}")
        
        # Row 5: Party Share Analysis
        col9, col10 = st.columns(2)
        
        with col9:
            # Party registration share pie chart
            fig_reg_share = px.pie(
                values=reg_values,
                names=parties,
                title="Registration Share by Party",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_reg_share.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_reg_share, use_container_width=True, key=f"party_reg_share_{dataset_key}")
        
        with col10:
            # Party vote share pie chart
            fig_vote_share = px.pie(
                values=vote_values,
                names=parties,
                title="Vote Share by Party",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_vote_share.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_vote_share, use_container_width=True, key=f"party_vote_share_{dataset_key}")
    
    else:
        # Row 3 alternative: If no party data, show engagement analysis
        st.subheader("📈 Engagement Analysis")
        col5, col6 = st.columns(2)
        
        with col5:
            # Simple engagement rate chart
            engagement_data = {
                'Category': ['Voted', 'Registered but Did Not Vote'],
                'Count': [stats['total_voted'], stats['registered_not_voted']],
                'Percentage': [
                    (stats['total_voted'] / stats['total_registered']) * 100,
                    (stats['registered_not_voted'] / stats['total_registered']) * 100
                ]
            }
            fig_engagement = px.bar(
                x=engagement_data['Category'],
                y=engagement_data['Count'],
                title='Voter Engagement Overview',
                color=engagement_data['Percentage'],
                color_continuous_scale='RdYlGn',
                text=engagement_data['Count']
            )
            fig_engagement.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_engagement, use_container_width=True, key=f"engagement_{dataset_key}")
        
        with col6:
            # Turnout efficiency by precinct size
            avg_registered_per_precinct = stats['total_registered'] / stats['total_rows']
            avg_voted_per_precinct = stats['total_voted'] / stats['total_rows']
            
            fig_efficiency = go.Figure()
            fig_efficiency.add_trace(go.Bar(
                name='Avg Registered per Precinct',
                x=['Average per Precinct'],
                y=[avg_registered_per_precinct],
                marker_color='lightblue',
                text=[f'{avg_registered_per_precinct:.0f}'],
                textposition='outside'
            ))
            fig_efficiency.add_trace(go.Bar(
                name='Avg Voted per Precinct',
                x=['Average per Precinct'],
                y=[avg_voted_per_precinct],
                marker_color='darkgreen',
                text=[f'{avg_voted_per_precinct:.0f}'],
                textposition='outside'
            ))
            
            fig_efficiency.update_layout(
                title='Average Engagement per Precinct',
                barmode='group',
                xaxis_title='Metric',
                yaxis_title='Count'
            )
            st.plotly_chart(fig_efficiency, use_container_width=True, key=f"efficiency_{dataset_key}")
    
    # Row 6: Additional Insights
    st.subheader("💡 Additional Insights")
    col11, col12 = st.columns(2)
    
    with col11:
        # Participation funnel
        # More realistic eligible population estimate
        estimated_eligible_population = int(stats['total_registered'] / 0.7)  # Assume 70% registration rate
        funnel_data = {
            'Stage': ['Eligible Population (Est.)', 'Registered Voters', 'Actual Voters'],
            'Count': [
                estimated_eligible_population,
                stats['total_registered'],
                stats['total_voted']
            ],
            'Color': ['lightgray', 'lightblue', 'darkgreen']
        }
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent initial",
            marker={"color": funnel_data['Color']}
        ))
        fig_funnel.update_layout(title="Voter Participation Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True, key=f"funnel_{dataset_key}")
    
    with col12:
        # Performance indicators
        indicators = {
            'Metric': ['Turnout Rate', 'Precinct Coverage', 'Avg Turnout/Precinct'],
            'Value': [
                stats['turnout_rate'],
                100,
                (stats['total_voted'] / stats['total_rows'])
            ],
            'Target': [70, 100, 500],
            'Status': []
        }
        
        for i, (val, target) in enumerate(zip(indicators['Value'], indicators['Target'])):
            if val >= target:
                indicators['Status'].append('Excellent')
            elif val >= target * 0.8:
                indicators['Status'].append('Good')
            elif val >= target * 0.6:
                indicators['Status'].append('Fair')
            else:
                indicators['Status'].append('Needs Improvement')
        
        colors = {'Excellent': 'green', 'Good': 'lightgreen', 'Fair': 'yellow', 'Needs Improvement': 'red'}
        fig_performance = px.bar(
            x=indicators['Metric'],
            y=indicators['Value'],
            title='Performance Dashboard',
            color=[colors[status] for status in indicators['Status']],
            text=[f'{val:.1f}' for val in indicators['Value']]
        )
        fig_performance.update_traces(textposition='outside')
        fig_performance.update_layout(showlegend=False)
        st.plotly_chart(fig_performance, use_container_width=True, key=f"performance_{dataset_key}")

    # Enhanced Analysis Sections
    
    # Row 7: Precinct Performance Analysis
    if stats.get('precinct_performance') is not None and len(stats['precinct_performance']) > 0:
        st.subheader("🏆 Precinct Performance Analysis")
        
        col13, col14 = st.columns(2)
        
        with col13:
            # Top performing precincts
            top_precincts = stats['precinct_performance'].head(10)
            if len(top_precincts) > 0:
                fig_top_precincts = px.bar(
                    top_precincts,
                    x=top_precincts.iloc[:, 0],  # Precinct name column
                    y='turnout_rate',
                    color='performance_tier',
                    title='Top 10 Performing Precincts',
                    labels={'x': 'Precinct', 'turnout_rate': 'Turnout Rate (%)'},
                    text='turnout_rate'
                )
                fig_top_precincts.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_top_precincts.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_top_precincts, use_container_width=True, key=f"top_precincts_{dataset_key}")
        
        with col14:
            # Performance distribution
            if len(stats['precinct_performance']) > 5:
                fig_distribution = px.histogram(
                    stats['precinct_performance'],
                    x='turnout_rate',
                    nbins=15,
                    title='Distribution of Precinct Turnout Rates',
                    labels={'turnout_rate': 'Turnout Rate (%)', 'count': 'Number of Precincts'},
                    color_discrete_sequence=['#3498db']
                )
                
                # Add benchmark lines
                fig_distribution.add_vline(x=50, line_dash="dash", line_color="red", 
                                         annotation_text="Average (50%)")
                fig_distribution.add_vline(x=65, line_dash="dash", line_color="green", 
                                         annotation_text="Good (65%)")
                st.plotly_chart(fig_distribution, use_container_width=True, key=f"distribution_{dataset_key}")
        
        # Performance summary
        if stats.get('hotspots'):
            col15, col16 = st.columns(2)
            
            with col15:
                st.markdown("**🎯 High Performers**")
                high_perf = stats['hotspots']['high_performers']
                st.metric("Count", high_perf['count'])
                st.metric("Avg Turnout", f"{high_perf['avg_turnout']:.1f}%")
                if high_perf['precincts']:
                    st.write("**Top Precincts:**")
                    for precinct in high_perf['precincts']:
                        st.write(f"• {precinct}")
            
            with col16:
                st.markdown("**⚠️ Need Attention**")
                low_perf = stats['hotspots']['low_performers']
                st.metric("Count", low_perf['count'])
                st.metric("Avg Turnout", f"{low_perf['avg_turnout']:.1f}%")
                if low_perf['precincts']:
                    st.write("**Focus Areas:**")
                    for precinct in low_perf['precincts']:
                        st.write(f"• {precinct}")
    
    # Row 8: Voting Methods Analysis
    if stats.get('voting_methods') and len(stats['voting_methods']) > 1:
        st.subheader("📮 Voting Method Analysis")
        
        col17, col18 = st.columns(2)
        
        with col17:
            # Method turnout rates
            methods = list(stats['voting_methods'].keys())
            method_rates = [stats['voting_methods'][method]['avg_turnout_rate'] for method in methods]
            
            fig_methods = px.bar(
                x=methods,
                y=method_rates,
                title='Turnout Rate by Voting Method',
                labels={'x': 'Voting Method', 'y': 'Turnout Rate (%)'},
                color=method_rates,
                color_continuous_scale='RdYlGn',
                text=[f'{rate:.1f}%' for rate in method_rates]
            )
            fig_methods.update_traces(textposition='outside')
            st.plotly_chart(fig_methods, use_container_width=True, key=f"methods_{dataset_key}")
        
        with col18:
            # Method volume comparison
            method_volumes = [stats['voting_methods'][method]['total_voted'] for method in methods]
            
            fig_volume = px.pie(
                values=method_volumes,
                names=methods,
                title='Vote Volume by Method',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_volume.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_volume, use_container_width=True, key=f"volume_{dataset_key}")
    
    # Row 9: Age Demographics Analysis
    if stats.get('age_demographics') and len(stats['age_demographics']) > 1:
        st.subheader("👥 Age Demographics Analysis")
        
        age_data = {k: v for k, v in stats['age_demographics'].items() if not k.startswith('_')}
        
        if age_data:
            col19, col20 = st.columns(2)
            
            with col19:
                # Age group turnout rates
                age_groups = list(age_data.keys())
                age_turnout_rates = [age_data[group]['turnout_rate'] for group in age_groups]
                
                fig_age_turnout = px.bar(
                    x=age_groups,
                    y=age_turnout_rates,
                    title='Turnout Rate by Age Group',
                    labels={'x': 'Age Group', 'y': 'Turnout Rate (%)'},
                    color=age_turnout_rates,
                    color_continuous_scale='RdYlGn',
                    text=[f'{rate:.1f}%' for rate in age_turnout_rates]
                )
                fig_age_turnout.update_traces(textposition='outside')
                fig_age_turnout.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_age_turnout, use_container_width=True, key=f"age_turnout_{dataset_key}")
            
            with col20:
                # Age group registration numbers
                age_reg_numbers = [age_data[group]['registered'] for group in age_groups]
                
                fig_age_reg = px.pie(
                    values=age_reg_numbers,
                    names=age_groups,
                    title='Registration Distribution by Age',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_age_reg.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_age_reg, use_container_width=True, key=f"age_reg_{dataset_key}")
            
            # Age summary metrics
            if stats['age_demographics'].get('_summary'):
                summary = stats['age_demographics']['_summary']
                col21, col22, col23 = st.columns(3)
                
                with col21:
                    st.metric("Age Data Coverage", f"{summary['coverage_percentage']:.1f}%", 
                             f"{summary['total_with_known_age']:,} voters")
                with col22:
                    if summary.get('avg_age'):
                        st.metric("Average Age", f"{summary['avg_age']:.1f} years")
                with col23:
                    if summary.get('median_age'):
                        st.metric("Median Age", f"{summary['median_age']:.1f} years")
    
    # Row 10: Efficiency & Benchmark Analysis
    if stats.get('efficiency_metrics') or stats.get('benchmarks'):
        st.subheader("📊 Efficiency & Benchmark Analysis")
        
        col26, col27 = st.columns(2)
        
        with col26:
            # Registration efficiency
            if stats.get('efficiency_metrics'):
                efficiency = stats['efficiency_metrics']
                
                efficiency_metrics = ['Registration Rate', 'Voting Rate (Eligible)', 'Voting Rate (Registered)']
                efficiency_values = [
                    efficiency.get('registration_rate', 0),
                    efficiency.get('voting_rate_of_eligible', 0),
                    efficiency.get('voting_rate_of_registered', 0)
                ]
                
                fig_efficiency = px.bar(
                    x=efficiency_metrics,
                    y=efficiency_values,
                    title='Registration & Voting Efficiency',
                    labels={'x': 'Metric', 'y': 'Rate (%)'},
                    color=efficiency_values,
                    color_continuous_scale='Blues',
                    text=[f'{val:.1f}%' for val in efficiency_values]
                )
                fig_efficiency.update_traces(textposition='outside')
                fig_efficiency.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_efficiency, use_container_width=True, key=f"efficiency_metrics_{dataset_key}")
        
        with col27:
            # Benchmark comparison
            if stats.get('benchmarks'):
                benchmarks = stats['benchmarks']
                current_turnout = stats['turnout_rate']
                
                benchmark_names = ['Presidential Avg', 'Good Turnout', 'Average Turnout', 'Midterm Avg']
                benchmark_keys = ['presidential_avg', 'good_turnout', 'average_turnout', 'midterm_avg']
                benchmark_values = [benchmarks[key]['benchmark_value'] for key in benchmark_keys if key in benchmarks]
                benchmark_names = [name for i, name in enumerate(benchmark_names) if benchmark_keys[i] in benchmarks]
                
                # Add current performance
                benchmark_names.append('Your Performance')
                benchmark_values.append(current_turnout)
                
                colors = ['lightgray'] * (len(benchmark_values) - 1) + ['#e74c3c']
                
                fig_benchmarks = px.bar(
                    x=benchmark_names,
                    y=benchmark_values,
                    title='Benchmark Comparison',
                    labels={'x': 'Benchmark', 'y': 'Turnout Rate (%)'},
                    color=colors,
                    text=[f'{val:.1f}%' for val in benchmark_values]
                )
                fig_benchmarks.update_traces(textposition='outside', marker_color=colors)
                fig_benchmarks.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_benchmarks, use_container_width=True, key=f"benchmarks_{dataset_key}")
    
    # Final Row: Actionable Insights
    st.subheader("💡 Actionable Insights & Recommendations")
    
    final_col1, final_col2 = st.columns(2)
    
    with final_col1:
        st.markdown("**🎯 Key Opportunities**")
        
        insights = []
        
        if stats.get('efficiency_metrics'):
            efficiency = stats['efficiency_metrics']
            if efficiency.get('potential_new_voters', 0) > 1000:
                insights.append(f"**Registration Gap**: {efficiency['potential_new_voters']:,} potential new voters could be registered")
            if efficiency.get('potential_turnout_improvement', 0) > 1000:
                insights.append(f"**Turnout Gap**: {efficiency['potential_turnout_improvement']:,} registered voters didn't participate")
        
        if stats.get('hotspots') and stats['hotspots'].get('low_performers'):
            low_count = stats['hotspots']['low_performers']['count']
            if low_count > 0:
                insights.append(f"**Focus Areas**: {low_count} precincts need targeted outreach")
        
        if stats.get('voting_methods') and len(stats['voting_methods']) > 1:
            best_method = max(stats['voting_methods'].items(), key=lambda x: x[1]['avg_turnout_rate'])
            insights.append(f"**Best Method**: {best_method[0]} has {best_method[1]['avg_turnout_rate']:.1f}% turnout")
        
        if stats.get('age_demographics'):
            age_data = {k: v for k, v in stats['age_demographics'].items() if not k.startswith('_')}
            if age_data:
                best_age_group = max(age_data.items(), key=lambda x: x[1]['turnout_rate'])
                worst_age_group = min(age_data.items(), key=lambda x: x[1]['turnout_rate'])
                insights.append(f"**Age Engagement**: {best_age_group[0]} has highest turnout ({best_age_group[1]['turnout_rate']:.1f}%)")
                insights.append(f"**Age Opportunity**: {worst_age_group[0]} needs attention ({worst_age_group[1]['turnout_rate']:.1f}% turnout)")
        
        if not insights:
            insights.append("Analysis complete - review charts above for detailed insights")
        
        for insight in insights:
            st.write(f"• {insight}")
    
    with final_col2:
        st.markdown("**📈 Performance Summary**")
        
        summary_items = []
        
        if stats.get('benchmarks'):
            good_benchmark = stats['benchmarks'].get('good_turnout')
            if good_benchmark:
                if good_benchmark['performance'] == 'Above':
                    summary_items.append("✅ **Above Good Turnout Threshold**")
                else:
                    gap = abs(good_benchmark['difference'])
                    summary_items.append(f"📊 **{gap:.1f}% below Good Turnout threshold**")
        
        if stats.get('precinct_performance') is not None:
            total_precincts = len(stats['precinct_performance'])
            if total_precincts > 0:
                try:
                    excellent_count = len(stats['precinct_performance'][stats['precinct_performance']['performance_tier'] == 'Excellent'])
                    excellent_pct = (excellent_count / total_precincts) * 100
                    summary_items.append(f"🏆 **{excellent_pct:.1f}% of precincts performing excellently**")
                except:
                    summary_items.append(f"🏆 **{total_precincts} precincts analyzed**")
        
        if stats['turnout_rate'] >= 70:
            summary_items.append("🎉 **Strong Overall Performance**")
        elif stats['turnout_rate'] >= 50:
            summary_items.append("👍 **Moderate Performance with Room for Growth**")
        else:
            summary_items.append("⚠️ **Significant Improvement Opportunities**")
        
        for item in summary_items:
            st.write(item)

def generate_report_data(stats):
    """Generate comprehensive report data for export"""
    report_data = {
        'dataset_name': stats['name'],
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'total_precincts': stats['total_rows'],
            'total_registered': stats['total_registered'],
            'total_voted': stats['total_voted'],
            'turnout_rate': round(stats['turnout_rate'], 2),
            'registered_not_voted': stats['registered_not_voted']
        },
        'data_sources': {
            'registration_column': stats['reg_column_used'],
            'voting_column': stats['vote_column_used'],
            'rows_filtered': stats.get('rows_filtered', 0)
        },
        'party_breakdown': stats.get('party_breakdown', {}),
        'debug_info': stats.get('debug_info', [])
    }
    return report_data

def export_to_html(datasets_stats):
    """Generate comprehensive HTML report with all interactive Plotly charts"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        import plotly.utils
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Voter Turnout Analysis Report</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 15px; margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .dataset-section {{ background: white; margin: 30px 0; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
        .dataset-title {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 25px; font-size: 1.8em; }}
        .section-title {{ color: #2c3e50; margin: 30px 0 20px 0; font-size: 1.4em; border-left: 4px solid #3498db; padding-left: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 25px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2.2em; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }}
        .chart-container {{ margin: 30px 0; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
        .chart-single {{ margin: 30px 0; }}
        .party-section {{ margin: 30px 0; }}
        .party-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .party-card {{ background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; }}
        .party-card h4 {{ margin: 0 0 10px 0; font-size: 1.2em; }}
        .party-value {{ font-size: 1.8em; font-weight: bold; margin: 10px 0; }}
        .party-details {{ opacity: 0.9; font-size: 0.9em; }}
        .data-sources {{ background: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .data-sources strong {{ color: #2c3e50; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; }}
        .summary-table th, .summary-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .summary-table th {{ background: #3498db; color: white; font-weight: 600; }}
        .summary-table tr:nth-child(even) {{ background: #f8f9fa; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #7f8c8d; }}
        .insights {{ background: #e8f5e8; border-left: 5px solid #27ae60; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .warning {{ background: #fdf2e9; border-left: 5px solid #e67e22; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .comparison-section {{ background: #f8f9fa; padding: 30px; border-radius: 15px; margin: 30px 0; }}
        .insights-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .insight-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db; }}
        .hotspot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .hotspot-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        @media (max-width: 768px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
            .insights-grid {{ grid-template-columns: 1fr; }}
            .hotspot-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗳️ Comprehensive Voter Turnout Analysis Report</h1>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p>Complete interactive analysis with all visualizations and insights</p>
        </div>
"""
        
        # Executive Summary for multiple datasets
        if len(datasets_stats) > 1:
            total_precincts = sum(stats['total_rows'] for stats in datasets_stats)
            total_registered = sum(stats['total_registered'] for stats in datasets_stats)
            total_voted = sum(stats['total_voted'] for stats in datasets_stats)
            avg_turnout = (total_voted / total_registered * 100) if total_registered > 0 else 0
            
            html_content += f"""
        <div class="dataset-section">
            <h2 class="dataset-title">📋 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(datasets_stats)}</div>
                    <div class="metric-label">Datasets Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_precincts:,}</div>
                    <div class="metric-label">Total Precincts</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_registered:,}</div>
                    <div class="metric-label">Total Registered</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_voted:,}</div>
                    <div class="metric-label">Total Voted</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_turnout:.1f}%</div>
                    <div class="metric-label">Average Turnout</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📊 Executive Overview Charts</h3>
                <div class="chart-row">
                    <div id="executive-turnout-chart"></div>
                    <div id="executive-precincts-chart"></div>
                </div>
            </div>
        </div>
"""
            
            # Create executive summary charts
            names = [stats['name'] for stats in datasets_stats]
            turnout_rates = [float(stats['turnout_rate']) for stats in datasets_stats]
            precinct_counts = [int(stats['total_rows']) for stats in datasets_stats]
            
            names_js = json.dumps(names)
            turnout_rates_js = json.dumps(turnout_rates)
            precinct_counts_js = json.dumps(precinct_counts)
            
            html_content += f"""
<script>
    // Executive turnout comparison
    var execTurnoutData = {{
        data: [{{
            x: {names_js},
            y: {turnout_rates_js},
            type: 'bar',
            text: {turnout_rates_js}.map(rate => rate.toFixed(1) + '%'),
            textposition: 'outside',
            marker: {{ 
                color: {turnout_rates_js},
                colorscale: 'RdYlGn',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Turnout Rate: %{{y:.1f}}%<extra></extra>'
        }}],
        layout: {{
            title: 'Turnout Rate Comparison Across Datasets',
            xaxis: {{ title: 'Dataset' }},
            yaxis: {{ title: 'Turnout Rate (%)', range: [0, Math.max(...{turnout_rates_js}) * 1.1] }},
            template: 'plotly_white',
            height: 400,
            margin: {{ t: 50, b: 50, l: 50, r: 50 }}
        }}
    }};
    Plotly.newPlot('executive-turnout-chart', execTurnoutData.data, execTurnoutData.layout, {{responsive: true}});
    
    // Executive precincts comparison
    var execPrecinctsData = {{
        data: [{{
            x: {names_js},
            y: {precinct_counts_js},
            type: 'bar',
            text: {precinct_counts_js}.map(count => count.toLocaleString()),
            textposition: 'outside',
            marker: {{ 
                color: {precinct_counts_js},
                colorscale: 'Blues',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Precincts: %{{y:,}}<extra></extra>'
        }}],
        layout: {{
            title: 'Total Precincts by Dataset',
            xaxis: {{ title: 'Dataset' }},
            yaxis: {{ title: 'Number of Precincts', range: [0, Math.max(...{precinct_counts_js}) * 1.1] }},
            template: 'plotly_white',
            height: 400,
            margin: {{ t: 50, b: 50, l: 50, r: 50 }}
        }}
    }};
    Plotly.newPlot('executive-precincts-chart', execPrecinctsData.data, execPrecinctsData.layout, {{responsive: true}});
</script>
"""
        
        # Individual dataset analysis with comprehensive charts
        for i, stats in enumerate(datasets_stats):
            html_content += f"""
        <div class="dataset-section">
            <h2 class="dataset-title">📊 Complete Analysis: {stats['name']}</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{stats['total_registered']:,}</div>
                    <div class="metric-label">Total Registered</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stats['total_voted']:,}</div>
                    <div class="metric-label">Total Voted</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stats['turnout_rate']:.2f}%</div>
                    <div class="metric-label">Turnout Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stats['total_rows']:,}</div>
                    <div class="metric-label">Total Precincts</div>
                </div>
            </div>
            
            <div class="data-sources">
                <strong>📊 Data Sources:</strong> Registration from '{stats['reg_column_used']}', Voting from '{stats['vote_column_used']}'<br>
                <strong>🔧 Processing:</strong> {stats.get('rows_filtered', 0)} summary/outlier rows filtered for accuracy
            </div>
"""
            
            # Comprehensive chart sections
            html_content += f"""
            <!-- Overview Analysis -->
            <div class="chart-container">
                <h3 class="section-title">📈 Overview Analysis</h3>
                <div class="chart-row">
                    <div id="turnout-pie-{i}"></div>
                    <div id="gauge-chart-{i}"></div>
                </div>
                <div class="chart-row">
                    <div id="key-metrics-{i}"></div>
                    <div id="donut-chart-{i}"></div>
                </div>
            </div>
            
            <!-- Engagement Analysis -->
            <div class="chart-container">
                <h3 class="section-title">🎯 Engagement Analysis</h3>
                <div class="chart-row">
                    <div id="engagement-chart-{i}"></div>
                    <div id="efficiency-chart-{i}"></div>
                </div>
            </div>
            
            <!-- Additional Insights -->
            <div class="chart-container">
                <h3 class="section-title">💡 Additional Insights</h3>
                <div class="chart-row">
                    <div id="funnel-chart-{i}"></div>
                    <div id="performance-chart-{i}"></div>
                </div>
            </div>
"""
            
            # Prepare comprehensive data
            voted_count = int(stats['total_voted'])
            not_voted_count = int(stats['registered_not_voted'])
            turnout_rate = float(stats['turnout_rate'])
            total_registered = int(stats['total_registered'])
            total_rows = int(stats['total_rows'])
            
            # Calculate additional metrics
            # More realistic registration efficiency calculation
            estimated_eligible_for_donut = int(total_registered / 0.7)  # Assume 70% registration rate
            reg_efficiency = (total_registered / estimated_eligible_for_donut) * 100
            non_registered = max(0, 100 - reg_efficiency)
            avg_registered_per_precinct = total_registered / total_rows
            avg_voted_per_precinct = voted_count / total_rows
            # Fixed eligible population calculation - more realistic estimate
            eligible_population = estimated_eligible_for_donut
            
            html_content += f"""
<script>
    // Comprehensive charts for dataset {i}
    
    // 1. Pie chart for turnout breakdown
    var pieData{i} = {{
        data: [{{
            values: [{voted_count}, {not_voted_count}],
            labels: ['Voted', 'Registered but Did Not Vote'],
            type: 'pie',
            textposition: 'inside',
            textinfo: 'percent+label',
            marker: {{
                colors: ['#2E8B57', '#FFD700']
            }},
            hovertemplate: '%{{label}}<br>Count: %{{value:,}}<br>Percentage: %{{percent}}<extra></extra>'
        }}],
        layout: {{
            title: 'Voter Turnout Breakdown - {stats["name"]}',
            template: 'plotly_white',
            height: 400,
            font: {{ size: 12 }}
        }}
    }};
    Plotly.newPlot('turnout-pie-{i}', pieData{i}.data, pieData{i}.layout, {{responsive: true}});
    
    // 2. Gauge chart for turnout rate
    var gaugeData{i} = {{
        data: [{{
            type: "indicator",
            mode: "gauge+number",
            value: {turnout_rate},
            title: {{ text: "Overall Turnout Rate (%)" }},
            gauge: {{
                axis: {{ range: [null, 100] }},
                bar: {{ color: "darkblue" }},
                steps: [
                    {{ range: [0, 30], color: "lightcoral" }},
                    {{ range: [30, 50], color: "lightyellow" }},
                    {{ range: [50, 70], color: "lightgreen" }},
                    {{ range: [70, 100], color: "darkgreen" }}
                ],
                threshold: {{
                    line: {{ color: "red", width: 4 }},
                    thickness: 0.75,
                    value: 90
                }}
            }}
        }}],
        layout: {{
            template: 'plotly_white',
            height: 400,
            font: {{ size: 12 }}
        }}
    }};
    Plotly.newPlot('gauge-chart-{i}', gaugeData{i}.data, gaugeData{i}.layout, {{responsive: true}});
    
    // 3. Key metrics bar chart
    var keyMetricsData{i} = {{
        data: [{{
            x: ['Total Precincts', 'Total Registered', 'Total Voted'],
            y: [{total_rows}, {total_registered}, {voted_count}],
            type: 'bar',
            text: ['{total_rows:,}', '{total_registered:,}', '{voted_count:,}'],
            textposition: 'outside',
            marker: {{
                color: [{total_rows}, {total_registered}, {voted_count}],
                colorscale: 'viridis',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Count: %{{y:,}}<extra></extra>'
        }}],
        layout: {{
            title: 'Key Metrics - {stats["name"]}',
            template: 'plotly_white',
            height: 400,
            showlegend: false
        }}
    }};
    Plotly.newPlot('key-metrics-{i}', keyMetricsData{i}.data, keyMetricsData{i}.layout, {{responsive: true}});
    
    // 4. Donut chart for registration efficiency
    var donutData{i} = {{
        data: [{{
            values: [{reg_efficiency:.1f}, {non_registered:.1f}],
            labels: ['Registered', 'Potentially Unregistered'],
            type: 'pie',
            hole: 0.5,
            textposition: 'inside',
            textinfo: 'percent+label',
            marker: {{
                colors: ['#1f77b4', '#d62728']
            }},
            hovertemplate: '%{{label}}<br>Percentage: %{{percent}}<extra></extra>'
        }}],
        layout: {{
            title: 'Registration Coverage Estimate',
            template: 'plotly_white',
            height: 400,
            annotations: [{{
                text: '{reg_efficiency:.1f}%',
                x: 0.5, y: 0.5,
                font_size: 20,
                showarrow: false
            }}]
        }}
    }};
    Plotly.newPlot('donut-chart-{i}', donutData{i}.data, donutData{i}.layout, {{responsive: true}});
    
    // 5. Engagement chart
    var engagementData{i} = {{
        data: [{{
            x: ['Voted', 'Registered but Did Not Vote'],
            y: [{voted_count}, {not_voted_count}],
            type: 'bar',
            text: ['{voted_count:,}', '{not_voted_count:,}'],
            textposition: 'outside',
            marker: {{
                color: [{(voted_count/total_registered)*100:.1f}, {(not_voted_count/total_registered)*100:.1f}],
                colorscale: 'RdYlGn',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Count: %{{y:,}}<extra></extra>'
        }}],
        layout: {{
            title: 'Voter Engagement Overview',
            xaxis: {{ title: 'Category' }},
            yaxis: {{ title: 'Count' }},
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('engagement-chart-{i}', engagementData{i}.data, engagementData{i}.layout, {{responsive: true}});
    
    // 6. Efficiency chart
    var efficiencyData{i} = {{
        data: [
            {{
                name: 'Avg Registered per Precinct',
                x: ['Average per Precinct'],
                y: [{avg_registered_per_precinct:.0f}],
                type: 'bar',
                marker: {{ color: 'lightblue' }},
                text: ['{avg_registered_per_precinct:.0f}'],
                textposition: 'outside',
                hovertemplate: 'Avg Registered: %{{y:.0f}}<extra></extra>'
            }},
            {{
                name: 'Avg Voted per Precinct',
                x: ['Average per Precinct'],
                y: [{avg_voted_per_precinct:.0f}],
                type: 'bar',
                marker: {{ color: 'darkgreen' }},
                text: ['{avg_voted_per_precinct:.0f}'],
                textposition: 'outside',
                hovertemplate: 'Avg Voted: %{{y:.0f}}<extra></extra>'
            }}
        ],
        layout: {{
            title: 'Average Engagement per Precinct',
            barmode: 'group',
            xaxis: {{ title: 'Metric' }},
            yaxis: {{ title: 'Count' }},
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('efficiency-chart-{i}', efficiencyData{i}.data, efficiencyData{i}.layout, {{responsive: true}});
    
    // 7. Participation funnel
    var funnelData{i} = {{
        data: [{{
            type: "funnel",
            y: ['Eligible Population (Est.)', 'Registered Voters', 'Actual Voters'],
            x: [{eligible_population}, {total_registered}, {voted_count}],
            textinfo: "value+percent initial",
            marker: {{ 
                color: ['lightgray', 'lightblue', 'darkgreen']
            }},
            hovertemplate: '%{{label}}<br>Count: %{{value:,}}<br>%{{percentInitial}} of eligible<extra></extra>'
        }}],
        layout: {{
            title: 'Voter Participation Funnel',
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('funnel-chart-{i}', funnelData{i}.data, funnelData{i}.layout, {{responsive: true}});
    
    // 8. Performance dashboard
    var performanceMetrics = ['Turnout Rate', 'Precinct Coverage', 'Avg Turnout/Precinct'];
    var performanceValues = [{turnout_rate}, 100, {avg_voted_per_precinct:.1f}];
    var performanceColors = [];
    
    // Determine colors based on performance
    performanceValues.forEach(function(val, idx) {{
        var target = idx === 0 ? 70 : (idx === 1 ? 100 : 500);
        if (val >= target) performanceColors.push('green');
        else if (val >= target * 0.8) performanceColors.push('lightgreen');
        else if (val >= target * 0.6) performanceColors.push('yellow');
        else performanceColors.push('red');
    }});
    
    var performanceData{i} = {{
        data: [{{
            x: performanceMetrics,
            y: performanceValues,
            type: 'bar',
            text: performanceValues.map(function(val, idx) {{
                return idx === 0 ? val.toFixed(1) + '%' : val.toFixed(1);
            }}),
            textposition: 'outside',
            marker: {{ color: performanceColors }},
            hovertemplate: '%{{x}}<br>Value: %{{text}}<extra></extra>'
        }}],
        layout: {{
            title: 'Performance Dashboard',
            xaxis: {{ title: 'Metric' }},
            yaxis: {{ title: 'Value' }},
            template: 'plotly_white',
            height: 400,
            showlegend: false
        }}
    }};
    Plotly.newPlot('performance-chart-{i}', performanceData{i}.data, performanceData{i}.layout, {{responsive: true}});
</script>
"""
            
            # Add party analysis if available
            if stats.get('party_breakdown'):
                html_content += f"""
            <!-- Party Analysis -->
            <div class="chart-container">
                <h3 class="section-title">🎭 Party Analysis</h3>
                <div class="chart-row">
                    <div id="party-registration-{i}"></div>
                    <div id="party-turnout-{i}"></div>
                </div>
                <div class="chart-row">
                    <div id="party-comparison-{i}"></div>
                    <div id="party-shares-{i}"></div>
                </div>
            </div>
            
            <div class="party-section">
                <div class="party-grid">
"""
                
                parties = list(stats['party_breakdown'].keys())
                reg_values = [int(stats['party_breakdown'][party]['registered']) for party in parties]
                vote_values = [int(stats['party_breakdown'][party]['voted']) for party in parties]
                party_turnout_rates = []
                
                for party, data in stats['party_breakdown'].items():
                    turnout_rate_party = (data['voted'] / data['registered'] * 100) if data['registered'] > 0 else 0
                    party_turnout_rates.append(round(turnout_rate_party, 2))
                    html_content += f"""
                <div class="party-card">
                    <h4>{party} Party</h4>
                    <div class="party-value">{turnout_rate_party:.1f}%</div>
                    <div class="party-details">
                        {data['voted']:,} voted out of {data['registered']:,} registered<br>
                        Registration: {(data['registered'] / stats['total_registered'] * 100):.1f}% of total
                    </div>
                </div>
"""
                
                html_content += """
                </div>
            </div>
"""
                
                # Party charts JavaScript
                parties_js = json.dumps(parties)
                reg_values_js = json.dumps(reg_values)
                vote_values_js = json.dumps(vote_values)
                turnout_rates_js = json.dumps(party_turnout_rates)
                
                html_content += f"""
<script>
    // Party analysis charts for dataset {i}
    
    // Party registration chart
    var partyRegData{i} = {{
        data: [{{
            x: {parties_js},
            y: {reg_values_js},
            type: 'bar',
            text: {reg_values_js}.map(val => val.toLocaleString()),
            textposition: 'outside',
            marker: {{ 
                color: {reg_values_js},
                colorscale: 'Blues',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Registered: %{{y:,}}<extra></extra>'
        }}],
        layout: {{
            title: 'Registration by Party',
            xaxis: {{ title: 'Party' }},
            yaxis: {{ title: 'Registered Voters' }},
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('party-registration-{i}', partyRegData{i}.data, partyRegData{i}.layout, {{responsive: true}});
    
    // Party turnout rates chart
    var partyTurnoutData{i} = {{
        data: [{{
            x: {parties_js},
            y: {turnout_rates_js},
            type: 'bar',
            text: {turnout_rates_js}.map(rate => rate.toFixed(1) + '%'),
            textposition: 'outside',
            marker: {{ 
                color: {turnout_rates_js},
                colorscale: 'RdYlGn',
                showscale: false
            }},
            hovertemplate: '%{{x}}<br>Turnout Rate: %{{y:.1f}}%<extra></extra>'
        }}],
        layout: {{
            title: 'Turnout Rate by Party (%)',
            xaxis: {{ title: 'Party' }},
            yaxis: {{ title: 'Turnout Rate (%)' }},
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('party-turnout-{i}', partyTurnoutData{i}.data, partyTurnoutData{i}.layout, {{responsive: true}});
    
    // Party comparison chart
    var partyCompData{i} = {{
        data: [
            {{
                name: 'Registered',
                x: {parties_js},
                y: {reg_values_js},
                type: 'bar',
                marker: {{ color: 'lightblue' }},
                text: {reg_values_js}.map(val => val.toLocaleString()),
                textposition: 'inside',
                hovertemplate: 'Registered<br>%{{x}}: %{{y:,}}<extra></extra>'
            }},
            {{
                name: 'Voted',
                x: {parties_js},
                y: {vote_values_js},
                type: 'bar',
                marker: {{ color: 'darkgreen' }},
                text: {vote_values_js}.map(val => val.toLocaleString()),
                textposition: 'inside',
                hovertemplate: 'Voted<br>%{{x}}: %{{y:,}}<extra></extra>'
            }}
        ],
        layout: {{
            title: 'Registration vs Voting by Party',
            barmode: 'group',
            xaxis: {{ title: 'Party' }},
            yaxis: {{ title: 'Count' }},
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('party-comparison-{i}', partyCompData{i}.data, partyCompData{i}.layout, {{responsive: true}});
    
    // Party share pie chart
    var partyShareData{i} = {{
        data: [{{
            values: {vote_values_js},
            labels: {parties_js},
            type: 'pie',
            textposition: 'inside',
            textinfo: 'percent+label',
            marker: {{
                colors: ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            }},
            hovertemplate: '%{{label}}<br>Votes: %{{value:,}}<br>Share: %{{percent}}<extra></extra>'
        }}],
        layout: {{
            title: 'Vote Share by Party',
            template: 'plotly_white',
            height: 400
        }}
    }};
    Plotly.newPlot('party-shares-{i}', partyShareData{i}.data, partyShareData{i}.layout, {{responsive: true}});
</script>
"""
            
            # Actionable insights section
            html_content += f"""
            <!-- Actionable Insights -->
            <div class="chart-container">
                <h3 class="section-title">💡 Actionable Insights & Recommendations</h3>
                <div class="insights-grid">
                    <div class="insight-card">
                        <h4>🎯 Key Opportunities</h4>
                        <ul>
"""
            
            # Generate insights
            insights = []
            if stats.get('efficiency_metrics'):
                efficiency = stats['efficiency_metrics']
                if efficiency.get('potential_new_voters', 0) > 1000:
                    insights.append(f"<strong>Registration Gap</strong>: {efficiency['potential_new_voters']:,} potential new voters could be registered")
                if efficiency.get('potential_turnout_improvement', 0) > 1000:
                    insights.append(f"<strong>Turnout Gap</strong>: {efficiency['potential_turnout_improvement']:,} registered voters didn't participate")
            
            avg_per_precinct = stats['total_voted'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
            # More realistic registration efficiency calculation  
            estimated_eligible_for_insights = int(stats['total_registered'] / 0.7)  # Assume 70% registration rate
            registration_efficiency = (stats['total_registered'] / estimated_eligible_for_insights) * 100
            
            insights.extend([
                f"<strong>Average turnout per precinct</strong>: {avg_per_precinct:.0f} voters",
                f"<strong>Registration efficiency</strong>: Approximately {registration_efficiency:.1f}% of eligible population registered",
                f"<strong>Participation gap</strong>: {stats['registered_not_voted']:,} registered voters did not participate"
            ])
            
            if stats.get('party_breakdown'):
                best_party = max(stats['party_breakdown'].items(), 
                               key=lambda x: (x[1]['voted']/x[1]['registered']) if x[1]['registered'] > 0 else 0)
                insights.append(f"<strong>Party engagement</strong>: {best_party[0]} party had the highest turnout rate")
            
            for insight in insights:
                html_content += f"<li>{insight}</li>"
            
            html_content += """
                        </ul>
                    </div>
                    <div class="insight-card">
                        <h4>📈 Performance Summary</h4>
                        <ul>
"""
            
            # Performance summary
            summary_items = []
            if stats['turnout_rate'] >= 70:
                summary_items.append("🎉 <strong>Strong Overall Performance</strong>")
            elif stats['turnout_rate'] >= 50:
                summary_items.append("👍 <strong>Moderate Performance with Room for Growth</strong>")
            else:
                summary_items.append("⚠️ <strong>Significant Improvement Opportunities</strong>")
            
            summary_items.extend([
                f"Overall turnout rate: {stats['turnout_rate']:.2f}%",
                f"Total precincts analyzed: {stats['total_rows']:,}",
                f"Voter participation rate: {(stats['total_voted']/eligible_population*100):.1f}% of eligible population"
            ])
            
            for item in summary_items:
                html_content += f"<li>{item}</li>"
            
            html_content += """
                        </ul>
                    </div>
                </div>
            </div>
"""
            
            # Performance analysis
            if stats['turnout_rate'] >= 70:
                performance_class = "insights"
                performance_icon = "✅"
                performance_text = "Excellent turnout rate! This indicates strong civic engagement."
            elif stats['turnout_rate'] >= 50:
                performance_class = "insights"  
                performance_icon = "👍"
                performance_text = "Good turnout rate, but there's room for improvement in voter engagement."
            else:
                performance_class = "warning"
                performance_icon = "⚠️"
                performance_text = "Low turnout rate indicates significant opportunity to increase voter participation."
            
            html_content += f"""
            <div class="{performance_class}">
                <strong>{performance_icon} Performance Analysis:</strong> {performance_text}
            </div>
        </div>
"""
        
        # Recommendations section
        html_content += """
        <div class="dataset-section">
            <h2 class="dataset-title">🎯 Recommendations for Improvement</h2>
            <div class="insights">
                <h4>📈 Strategies to Increase Voter Participation:</h4>
                <ol>
                    <li><strong>Targeted Outreach:</strong> Focus on precincts with below-average turnout rates</li>
                    <li><strong>Registration Drives:</strong> Increase voter registration in underperforming areas</li>
                    <li><strong>Education Campaigns:</strong> Inform voters about the importance and process of voting</li>
                    <li><strong>Accessibility Improvements:</strong> Ensure voting locations are convenient and accessible</li>
                    <li><strong>Early Voting & Mail-in Options:</strong> Provide multiple ways for people to cast their ballots</li>
                    <li><strong>Data-Driven Targeting:</strong> Use this analysis to identify specific improvement opportunities</li>
                    <li><strong>Community Partnerships:</strong> Work with local organizations to boost engagement</li>
                    <li><strong>Technology Integration:</strong> Leverage digital tools for voter education and engagement</li>
                </ol>
            </div>
        </div>
"""
        
        html_content += f"""
        <div class="footer">
            <p><strong>Comprehensive Interactive Voter Turnout Analysis Report</strong></p>
            <p>This report includes all visualizations and analysis from the Streamlit application</p>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p><em>Hover over charts for detailed information • Click legend items to show/hide data series</em></p>
            <p><strong>Charts included:</strong> Overview, Engagement, Party Analysis, Performance Dashboard, Funnel Analysis, and Actionable Insights</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
        
    except Exception as e:
        return f"""<!DOCTYPE html>
<html><head><title>Error</title></head>
<body>
<h1>Error generating comprehensive report</h1>
<p>Error details: {str(e)}</p>
<p>Please check that all required data is available and try again.</p>
</body></html>"""

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
                export_data.append(generate_report_data(stats))
            
            json_data = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="💾 Download JSON Report",
                data=json_data,
                file_name=f"voter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        # Export to HTML
        if st.button("🌐 Export to Interactive HTML", help="Export as interactive HTML report with charts"):
            try:
                with st.spinner("Generating interactive HTML report..."):
                    html_content = export_to_html(datasets_stats)
                
                st.download_button(
                    label="💾 Download Interactive HTML Report",
                    data=html_content,
                    file_name=f"voter_analysis_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.success("✅ Interactive HTML report generated successfully!")
                st.info("💡 The exported HTML file contains all the interactive charts and can be opened in any web browser.")
                
            except Exception as e:
                st.error(f"❌ Error generating interactive HTML: {str(e)}")
                st.write("Debug info:", str(datasets_stats[0].keys()) if datasets_stats else "No datasets")
    
    with col3:
        # Export to CSV summary
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
                
                # Add party data if available
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
    
    fig_comp3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Registered', 'Total Voted'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_comp3.add_trace(
        go.Bar(x=names, y=total_registered, name="Total Registered", marker_color='lightblue'),
        row=1, col=1
    )
    
    fig_comp3.add_trace(
        go.Bar(x=names, y=total_voted, name="Total Voted", marker_color='darkgreen'),
        row=1, col=2
    )
    
    fig_comp3.update_layout(title_text="Registration vs Turnout Comparison", showlegend=False)
    st.plotly_chart(fig_comp3, use_container_width=True, key="comparison_reg_vs_turnout")
    
    comparison_df = pd.DataFrame({
        'Dataset': names,
        'Total Precincts': total_rows,
        'Total Registered': total_registered,
        'Total Voted': total_voted,
        'Turnout Rate (%)': [f"{rate:.2f}" for rate in turnout_rates]
    })
    
    st.subheader("📋 Comparison Summary")
    st.dataframe(comparison_df, use_container_width=True)

# Login Interface
# Updated login method for newer streamlit-authenticator versions
try:
    authenticator.login()
    name = st.session_state.get("name")
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")
except Exception as e:
    # Fallback for different versions
    name = None
    authentication_status = None
    username = None

st.session_state['name'] = name
st.session_state['authentication_status'] = authentication_status
st.session_state['username'] = username

# Main Application
if st.session_state['authentication_status']:
    authenticator.logout("Logout", "sidebar")
    
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
                    
                    try:
                        models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"]
                        
                        response = None
                        for model in models_to_try:
                            try:
                                response = openai.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a civic engagement expert specializing in voter turnout analysis."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1000
                                )
                                break
                            except Exception as model_error:
                                if "model_not_found" in str(model_error):
                                    continue
                                else:
                                    raise model_error
                        
                        if response:
                            suggestions = response.choices[0].message.content
                            st.markdown("### 🤖 AI-Generated Improvement Suggestions")
                            st.write(suggestions)
                        else:
                            st.error("Unable to access AI models. Please check your API configuration.")
                    
                    except Exception as e:
                        st.error(f"AI request failed: {e}")
        
        if len(st.session_state['datasets']) > 1:
            datasets_stats = [info['stats'] for info in st.session_state['datasets'].values()]
            create_comparison_charts(datasets_stats)
        
        # Add export functionality
        datasets_stats = [info['stats'] for info in st.session_state['datasets'].values()]
        create_export_section(datasets_stats)
    
    else:
        st.info("👆 Upload CSV files to begin analysis")

elif st.session_state['authentication_status'] is False:
    st.error("❌ Username/password is incorrect")
elif st.session_state['authentication_status'] is None:
    st.warning("🔐 Please enter your username and password")
