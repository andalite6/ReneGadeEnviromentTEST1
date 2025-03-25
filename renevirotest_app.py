import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import logging
import os
import threading
import asyncio
import random
import base64
import traceback
from datetime import datetime, timedelta
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redteam_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RedTeamApp")

# Set page configuration with custom theme
st.set_page_config(
    page_title="AI Security Red Team Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state with error handling
def initialize_session_state():
    """Initialize all session state variables with proper error handling"""
    try:
        # Core session states
        if 'targets' not in st.session_state:
            st.session_state.targets = []

        if 'test_results' not in st.session_state:
            st.session_state.test_results = {}

        if 'running_test' not in st.session_state:
            st.session_state.running_test = False

        if 'progress' not in st.session_state:
            st.session_state.progress = 0

        if 'vulnerabilities_found' not in st.session_state:
            st.session_state.vulnerabilities_found = 0

        if 'current_theme' not in st.session_state:
            st.session_state.current_theme = "dark"  # Default to dark theme
            
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
            
        # Thread management
        if 'active_threads' not in st.session_state:
            st.session_state.active_threads = []
            
        # Error handling
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
            
        # Initialize bias testing state
        if 'bias_results' not in st.session_state:
            st.session_state.bias_results = {}
            
        if 'show_bias_results' not in st.session_state:
            st.session_state.show_bias_results = False
            
        # Carbon tracking states
        if 'carbon_tracking_active' not in st.session_state:
            st.session_state.carbon_tracking_active = False
            
        if 'carbon_measurements' not in st.session_state:
            st.session_state.carbon_measurements = []
            
        logger.info("Session state initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        display_error(f"Failed to initialize application state: {str(e)}")

# Thread cleanup
def cleanup_threads():
    """Remove completed threads from session state"""
    try:
        if 'active_threads' in st.session_state:
            # Filter out completed threads
            active_threads = []
            for thread in st.session_state.active_threads:
                if thread.is_alive():
                    active_threads.append(thread)
            
            # Update session state with only active threads
            st.session_state.active_threads = active_threads
            
            if len(st.session_state.active_threads) > 0:
                logger.info(f"Active threads: {len(st.session_state.active_threads)}")
    except Exception as e:
        logger.error(f"Error cleaning up threads: {str(e)}")

# Define color schemes
themes = {
    "dark": {
        "bg_color": "#121212",
        "card_bg": "#1E1E1E",
        "primary": "#1DB954",    # Vibrant green
        "secondary": "#BB86FC",  # Purple
        "accent": "#03DAC6",     # Teal
        "warning": "#FF9800",    # Orange
        "error": "#CF6679",      # Red
        "text": "#FFFFFF"
    },
    "light": {
        "bg_color": "#F5F5F5",
        "card_bg": "#FFFFFF",
        "primary": "#1DB954",    # Vibrant green
        "secondary": "#7C4DFF",  # Deep purple
        "accent": "#00BCD4",     # Cyan
        "warning": "#FF9800",    # Orange
        "error": "#F44336",      # Red
        "text": "#212121"
    }
}

# Get current theme colors safely
def get_theme():
    """Get current theme with error handling"""
    try:
        return themes[st.session_state.current_theme]
    except Exception as e:
        logger.error(f"Error getting theme: {str(e)}")
        # Return dark theme as fallback
        return themes["dark"]

# CSS styles
def load_css():
    """Load CSS with the current theme"""
    try:
        theme = get_theme()
        
        return f"""
        <style>
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["primary"]};
        }}
        
        .stProgress > div > div > div > div {{
            background-color: {theme["primary"]};
        }}
        
        div[data-testid="stExpander"] {{
            border: none;
            border-radius: 8px;
            background-color: {theme["card_bg"]};
            margin-bottom: 1rem;
        }}
        
        div[data-testid="stVerticalBlock"] {{
            gap: 1.5rem;
        }}
        
        .card {{
            border-radius: 10px;
            background-color: {theme["card_bg"]};
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 3px solid {theme["primary"]};
        }}
        
        .warning-card {{
            border-left: 3px solid {theme["warning"]};
        }}
        
        .error-card {{
            border-left: 3px solid {theme["error"]};
        }}
        
        .success-card {{
            border-left: 3px solid {theme["primary"]};
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: {theme["primary"]};
        }}
        
        .metric-label {{
            font-size: 14px;
            color: {theme["text"]};
            opacity: 0.7;
        }}
        
        .sidebar-title {{
            margin-left: 15px;
            font-size: 1.2rem;
            font-weight: bold;
            color: {theme["primary"]};
        }}
        
        .target-card {{
            border-radius: 8px;
            background-color: {theme["card_bg"]};
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid {theme["secondary"]};
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .status-badge.active {{
            background-color: {theme["primary"]};
            color: white;
        }}
        
        .status-badge.inactive {{
            background-color: gray;
            color: white;
        }}
        
        .hover-card:hover {{
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }}
        
        .card-title {{
            color: {theme["primary"]};
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .nav-item {{
            padding: 8px 15px;
            border-radius: 5px;
            margin-bottom: 5px;
            cursor: pointer;
        }}
        
        .nav-item:hover {{
            background-color: rgba(29, 185, 84, 0.1);
        }}
        
        .nav-item.active {{
            background-color: rgba(29, 185, 84, 0.2);
            font-weight: bold;
        }}
        
        .tag {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        .tag.owasp {{
            background-color: rgba(187, 134, 252, 0.2);
            color: {theme["secondary"]};
        }}
        
        .tag.nist {{
            background-color: rgba(3, 218, 198, 0.2);
            color: {theme["accent"]};
        }}
        
        .tag.fairness {{
            background-color: rgba(255, 152, 0, 0.2);
            color: {theme["warning"]};
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            border-radius: 5px 5px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme["card_bg"]};
            border-bottom: 3px solid {theme["primary"]};
        }}
        
        .error-message {{
            background-color: #CF6679;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        </style>
        """
    except Exception as e:
        logger.error(f"Error loading CSS: {str(e)}")
        # Return minimal CSS as fallback
        return "<style>.error-message { background-color: #CF6679; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; }</style>"

# Helper function to set page
def set_page(page_name):
    """Set the current page safely"""
    try:
        st.session_state.current_page = page_name
        logger.info(f"Navigation: Switched to {page_name} page")
    except Exception as e:
        logger.error(f"Error setting page to {page_name}: {str(e)}")
        display_error(f"Failed to navigate to {page_name}")

# Safe rerun function
def safe_rerun():
    """Safely rerun the app, handling different Streamlit versions"""
    try:
        st.rerun()  # For newer Streamlit versions
    except Exception as e1:
        try:
            st.experimental_rerun()  # For older Streamlit versions
        except Exception as e2:
            logger.error(f"Failed to rerun app: {str(e1)} then {str(e2)}")
            # Do nothing - at this point we can't fix it

# Error handling
def display_error(message):
    """Display error message to the user"""
    try:
        st.session_state.error_message = message
        logger.error(f"UI Error: {message}")
    except Exception as e:
        logger.critical(f"Failed to display error message: {str(e)}")

# Custom components
def card(title, content, card_type="default"):
    """Generate HTML card with error handling"""
    try:
        card_class = "card"
        if card_type == "warning":
            card_class += " warning-card"
        elif card_type == "error":
            card_class += " error-card"
        elif card_type == "success":
            card_class += " success-card"
        
        return f"""
        <div class="{card_class} hover-card">
            <div class="card-title">{title}</div>
            {content}
        </div>
        """
    except Exception as e:
        logger.error(f"Error rendering card: {str(e)}")
        return f"""
        <div class="card error-card">
            <div class="card-title">Error Rendering Card</div>
            <p>Failed to render card content: {str(e)}</p>
        </div>
        """

def metric_card(label, value, description="", prefix="", suffix=""):
    """Generate HTML metric card with error handling"""
    try:
        return f"""
        <div class="card hover-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
            <div style="font-size: 14px; opacity: 0.7;">{description}</div>
        </div>
        """
    except Exception as e:
        logger.error(f"Error rendering metric card: {str(e)}")
        return f"""
        <div class="card error-card">
            <div class="metric-label">Error</div>
            <div class="metric-value">N/A</div>
            <div style="font-size: 14px; opacity: 0.7;">Failed to render metric: {str(e)}</div>
        </div>
        """

# Logo and header
def render_header():
    """Render the application header safely"""
    try:
        logo_html = """
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="margin-right: 10px; font-size: 2.5rem;">🛡️</div>
            <div>
                <h1 style="margin-bottom: 0;">Synthetic Red Team Testing Agent</h1>
                <p style="opacity: 0.7;">Advanced Security Testing for AI Systems</p>
            </div>
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering header: {str(e)}")
        st.markdown("# 🛡️ Synthetic Red Team Testing Agent")

# Sidebar navigation - Fixed implementation
def sidebar_navigation():
    """Render the sidebar navigation with proper Streamlit buttons"""
    try:
        st.sidebar.markdown('<div class="sidebar-title">🧭 Navigation</div>', unsafe_allow_html=True)
        
        navigation_options = [
            {"icon": "🏠", "name": "Dashboard"},
            {"icon": "🎯", "name": "Target Management"},
            {"icon": "🧪", "name": "Test Configuration"},
            {"icon": "▶️", "name": "Run Assessment"},
            {"icon": "📊", "name": "Results Analyzer"},
            {"icon": "🔍", "name": "Ethical AI Testing"},
            {"icon": "🌱", "name": "Environmental Impact"},  # New page
            {"icon": "⚖️", "name": "Bias Testing"},          # New page
            {"icon": "📁", "name": "Multi-Format Import"},   # New page
            {"icon": "🚀", "name": "High-Volume Testing"},
            {"icon": "⚙️", "name": "Settings"}
        ]
        
        for option in navigation_options:
            # Create a button for each navigation option
            if st.sidebar.button(
                f"{option['icon']} {option['name']}", 
                key=f"nav_{option['name']}",
                use_container_width=True,
                type="secondary" if st.session_state.current_page != option["name"] else "primary"
            ):
                set_page(option["name"])
                safe_rerun()
        
        # Theme toggle
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="sidebar-title">🎨 Appearance</div>', unsafe_allow_html=True)
        if st.sidebar.button("🔄 Toggle Theme", key="toggle_theme", use_container_width=True):
            st.session_state.current_theme = "light" if st.session_state.current_theme == "dark" else "dark"
            logger.info(f"Theme toggled to {st.session_state.current_theme}")
            safe_rerun()
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="sidebar-title">📡 System Status</div>', unsafe_allow_html=True)
        
        if st.session_state.running_test:
            st.sidebar.success("⚡ Test Running")
        else:
            st.sidebar.info("⏸️ Idle")
        
        st.sidebar.markdown(f"🎯 Targets: {len(st.session_state.targets)}")
        
        # Active threads info
        if len(st.session_state.active_threads) > 0:
            st.sidebar.markdown(f"🧵 Active threads: {len(st.session_state.active_threads)}")
        
        # Add carbon tracking status if active
        if st.session_state.get("carbon_tracking_active", False):
            st.sidebar.markdown("🌱 Carbon tracking active")
        
        # Add version info
        st.sidebar.markdown("---")
        st.sidebar.markdown("v1.0.0 | © 2025", unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering sidebar: {str(e)}")
        st.sidebar.error("Navigation Error")
        st.sidebar.markdown(f"Error: {str(e)}")

# Mock data functions with error handling
def get_mock_test_vectors():
    """Get mock test vector data with error handling"""
    try:
        return [
            {
                "id": "sql_injection",
                "name": "SQL Injection",
                "category": "owasp",
                "severity": "high"
            },
            {
                "id": "xss",
                "name": "Cross-Site Scripting",
                "category": "owasp",
                "severity": "medium"
            },
            {
                "id": "prompt_injection",
                "name": "Prompt Injection",
                "category": "owasp",
                "severity": "critical"
            },
            {
                "id": "insecure_output",
                "name": "Insecure Output Handling",
                "category": "owasp",
                "severity": "high"
            },
            {
                "id": "nist_governance",
                "name": "AI Governance",
                "category": "nist",
                "severity": "medium"
            },
            {
                "id": "nist_transparency",
                "name": "Transparency",
                "category": "nist",
                "severity": "medium"
            },
            {
                "id": "fairness_demographic",
                "name": "Demographic Parity",
                "category": "fairness",
                "severity": "high"
            },
            {
                "id": "privacy_gdpr",
                "name": "GDPR Compliance",
                "category": "privacy",
                "severity": "critical"
            },
            {
                "id": "jailbreaking",
                "name": "Jailbreaking Resistance",
                "category": "exploit",
                "severity": "critical"
            }
        ]
    except Exception as e:
        logger.error(f"Error getting mock test vectors: {str(e)}")
        display_error("Failed to load test vectors")
        return []  # Return empty list as fallback

def run_mock_test(target, test_vectors, duration=30):
    """Simulate running a test in the background with proper error handling"""
    try:
        # Initialize progress
        st.session_state.progress = 0
        st.session_state.vulnerabilities_found = 0
        st.session_state.running_test = True
        
        logger.info(f"Starting mock test against {target['name']} with {len(test_vectors)} test vectors")
        
        # Create mock results data structure
        results = {
            "summary": {
                "total_tests": 0,
                "vulnerabilities_found": 0,
                "risk_score": 0
            },
            "vulnerabilities": [],
            "test_details": {}
        }
        
        # Simulate test execution
        total_steps = 100
        step_sleep = duration / total_steps
        
        for i in range(total_steps):
            # Check if we should stop (for handling cancellations)
            if not st.session_state.running_test:
                logger.info("Test was cancelled")
                break
                
            time.sleep(step_sleep)
            st.session_state.progress = (i + 1) / total_steps
            
            # Occasionally "find" a vulnerability
            if random.random() < 0.2:  # 20% chance each step
                vector = random.choice(test_vectors)
                severity_weight = {"low": 1, "medium": 2, "high": 3, "critical": 5}
                weight = severity_weight.get(vector["severity"], 1)
                
                # Add vulnerability to results
                vulnerability = {
                    "id": f"VULN-{len(results['vulnerabilities']) + 1}",
                    "test_vector": vector["id"],
                    "test_name": vector["name"],
                    "severity": vector["severity"],
                    "details": f"Mock vulnerability found in {target['name']} using {vector['name']} test vector.",
                    "timestamp": datetime.now().isoformat()
                }
                results["vulnerabilities"].append(vulnerability)
                
                # Update counters
                st.session_state.vulnerabilities_found += 1
                results["summary"]["vulnerabilities_found"] += 1
                results["summary"]["risk_score"] += weight
                
                logger.info(f"Found vulnerability: {vulnerability['id']} ({vulnerability['severity']})")
        
        # Complete the test results
        results["summary"]["total_tests"] = len(test_vectors) * 10  # Assume 10 variations per vector
        results["timestamp"] = datetime.now().isoformat()
        results["target"] = target["name"]
        
        logger.info(f"Test completed: {results['summary']['vulnerabilities_found']} vulnerabilities found")
        
        # Set the results in session state
        st.session_state.test_results = results
        return results
    
    except Exception as e:
        error_details = {
            "error": True,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        logger.error(f"Error in test execution: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Create error result
        st.session_state.error_message = f"Test execution failed: {str(e)}"
        return error_details
    
    finally:
        # Always ensure we reset the running state
        st.session_state.running_test = False

# ================= NEW FUNCTIONALITY =================

# File Format Support Functions
def handle_multiple_file_formats(uploaded_file):
    """Process different file formats for impact assessments"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # JSON (already supported)
        if file_extension == 'json':
            import json
            return json.loads(uploaded_file.read())
        
        # CSV
        elif file_extension == 'csv':
            import pandas as pd
            import io
            return pd.read_csv(uploaded_file)
        
        # Excel
        elif file_extension in ['xlsx', 'xls']:
            import pandas as pd
            return pd.read_excel(uploaded_file)
        
        # PDF
        elif file_extension == 'pdf':
            from pypdf import PdfReader
            import io
            
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return {"text": text}
        
        # XML
        elif file_extension == 'xml':
            import xml.etree.ElementTree as ET
            import io
            
            tree = ET.parse(io.BytesIO(uploaded_file.read()))
            root = tree.getroot()
            
            # Convert XML to dict (simplified)
            def xml_to_dict(element):
                result = {}
                for child in element:
                    child_data = xml_to_dict(child)
                    if child.tag in result:
                        if type(result[child.tag]) is list:
                            result[child.tag].append(child_data)
                        else:
                            result[child.tag] = [result[child.tag], child_data]
                    else:
                        result[child.tag] = child_data
                
                if len(result) == 0:
                    return element.text
                return result
            
            return xml_to_dict(root)
        
        # YAML/YML
        elif file_extension in ['yaml', 'yml']:
            import yaml
            return yaml.safe_load(uploaded_file)
        
        # Parquet
        elif file_extension == 'parquet':
            import pandas as pd
            return pd.read_parquet(uploaded_file)
        
        # HDF5
        elif file_extension == 'h5':
            import h5py
            import io
            import numpy as np
            import json
            
            # Save uploaded file to a temporary file
            f = io.BytesIO(uploaded_file.read())
            h5_file = h5py.File(f, 'r')
            
            # Convert h5 to dict (simplified)
            def h5_to_dict(h5_file):
                result = {}
                for key in h5_file.keys():
                    if isinstance(h5_file[key], h5py.Dataset):
                        result[key] = h5_file[key][()].tolist() if isinstance(h5_file[key][()], (np.ndarray, list, tuple)) else h5_file[key][()]
                    else:
                        result[key] = h5_to_dict(h5_file[key])
                return result
            
            return h5_to_dict(h5_file)
        
        # Arrow
        elif file_extension == 'arrow':
            import pyarrow as pa
            import io
            
            f = io.BytesIO(uploaded_file.read())
            reader = pa.ipc.open_file(f)
            table = reader.read_all()
            
            return table.to_pandas()
        
        # JSONL
        elif file_extension == 'jsonl':
            import json
            import io
            
            lines = uploaded_file.read().decode('utf-8').splitlines()
            return [json.loads(line) for line in lines]
        
        else:
            return {"error": f"Unsupported file format: {file_extension}"}
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {"error": f"Failed to process file: {str(e)}"}


# WhyLabs Bias Testing Module
class WhyLabsBiasTest:
    """Class for WhyLabs-based bias testing functionality"""
    
    def __init__(self):
        import whylogs as why
        import pandas as pd
        
        self.why = why
        self.session = None
        self.results = {}
    
    def initialize_session(self, dataset_name):
        """Initialize a WhyLogs profiling session"""
        try:
            self.session = self.why.get_or_create_session()
            logger.info(f"WhyLogs session initialized for {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WhyLogs session: {str(e)}")
            return False
    
    def profile_dataset(self, df, dataset_name):
        """Profile a dataset for bias analysis"""
        try:
            if self.session is None:
                self.initialize_session(dataset_name)
                
            # Create a profile
            profile = self.session.log_dataframe(df, dataset_name=dataset_name)
            self.results[dataset_name] = {"profile": profile}
            logger.info(f"Dataset {dataset_name} profiled successfully")
            return profile
        except Exception as e:
            logger.error(f"Failed to profile dataset: {str(e)}")
            return None
    
    def analyze_bias(self, df, protected_features, target_column, dataset_name):
        """Analyze bias in a dataset based on protected features"""
        try:
            # Profile the dataset first
            profile = self.profile_dataset(df, dataset_name)
            
            bias_metrics = {}
            
            # Calculate basic bias metrics
            for feature in protected_features:
                # Statistical parity difference
                feature_groups = df.groupby(feature)
                
                outcomes = {}
                disparities = {}
                
                for group_name, group_data in feature_groups:
                    # For binary target variable
                    if df[target_column].nunique() == 2:
                        positive_outcome_rate = group_data[target_column].mean()
                        outcomes[group_name] = positive_outcome_rate
                
                # Calculate disparities between groups
                baseline = max(outcomes.values())
                for group, rate in outcomes.items():
                    disparities[group] = baseline - rate
                
                bias_metrics[feature] = {
                    "outcomes": outcomes,
                    "disparities": disparities,
                    "max_disparity": max(disparities.values())
                }
            
            self.results[dataset_name]["bias_metrics"] = bias_metrics
            logger.info(f"Bias analysis completed for {dataset_name}")
            return bias_metrics
        except Exception as e:
            logger.error(f"Failed to analyze bias: {str(e)}")
            return {"error": str(e)}
    
    def get_results(self, dataset_name=None):
        """Get analysis results"""
        if dataset_name:
            return self.results.get(dataset_name, {})
        return self.results


# Environmental Impact Assessment Module using CodeCarbon
class CarbonImpactTracker:
    """Class for tracking environmental impact of AI systems"""
    
    def __init__(self):
        # Placeholder for codecarbon import
        self.tracker = None
        self.measurements = []
        self.total_emissions = 0.0
        self.is_tracking = False
    
    def initialize_tracker(self, project_name, api_endpoint=None):
        """Initialize the carbon tracker"""
        try:
            # Import codecarbon (assumed to be installed)
            from codecarbon import EmissionsTracker
            
            self.tracker = EmissionsTracker(
                project_name=project_name,
                output_dir="./emissions",
                log_level="error",
                save_to_file=True
            )
            
            logger.info(f"Carbon tracker initialized for {project_name}")
            return True
        except ImportError:
            logger.error("CodeCarbon not installed. Installing...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "codecarbon"])
                from codecarbon import EmissionsTracker
                
                self.tracker = EmissionsTracker(
                    project_name=project_name,
                    output_dir="./emissions",
                    log_level="error",
                    save_to_file=True
                )
                
                logger.info(f"Carbon tracker initialized for {project_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to install CodeCarbon: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize carbon tracker: {str(e)}")
            return False
    
    def start_tracking(self):
        """Start tracking carbon emissions"""
        try:
            if self.tracker is None:
                return False
                
            self.tracker.start()
            self.is_tracking = True
            logger.info("Carbon emission tracking started")
            return True
        except Exception as e:
            logger.error(f"Failed to start carbon tracking: {str(e)}")
            return False
    
    def stop_tracking(self):
        """Stop tracking and get the emissions data"""
        try:
            if not self.is_tracking or self.tracker is None:
                return 0.0
                
            emissions = self.tracker.stop()
            self.is_tracking = False
            self.measurements.append(emissions)
            self.total_emissions += emissions
            
            logger.info(f"Carbon emission tracking stopped. Measured: {emissions} kg CO2eq")
            return emissions
        except Exception as e:
            logger.error(f"Failed to stop carbon tracking: {str(e)}")
            return 0.0
    
    def get_total_emissions(self):
        """Get total emissions tracked so far"""
        return self.total_emissions
    
    def get_all_measurements(self):
        """Get all measurements"""
        return self.measurements
    
    def generate_report(self):
        """Generate a report of carbon emissions"""
        try:
            energy_solutions = [
                {
                    "name": "Optimize AI Model Size",
                    "description": "Reduce model parameters and optimize architecture",
                    "potential_savings": "20-60% reduction in emissions",
                    "implementation_difficulty": "Medium"
                },
                {
                    "name": "Implement Model Distillation",
                    "description": "Create smaller, efficient versions of larger models",
                    "potential_savings": "40-80% reduction in emissions",
                    "implementation_difficulty": "High"
                },
                {
                    "name": "Use Efficient Hardware",
                    "description": "Deploy on energy-efficient hardware (e.g., specialized AI chips)",
                    "potential_savings": "30-50% reduction in emissions",
                    "implementation_difficulty": "Medium"
                },
                {
                    "name": "Carbon-Aware Deployment",
                    "description": "Schedule compute-intensive tasks during low-carbon intensity periods",
                    "potential_savings": "15-40% reduction in emissions",
                    "implementation_difficulty": "Low"
                },
                {
                    "name": "Renewable Energy Sources",
                    "description": "Deploy AI systems in data centers powered by renewable energy",
                    "potential_savings": "Up to 100% reduction in emissions",
                    "implementation_difficulty": "Medium"
                }
            ]
            
            # Calculate the impact
            kwh_per_kg_co2 = 0.6  # Approximate conversion factor
            energy_consumption = self.total_emissions / kwh_per_kg_co2
            
            trees_equivalent = self.total_emissions * 16.5  # Each kg CO2 ~ 16.5 trees for 1 day
            
            return {
                "total_emissions_kg": self.total_emissions,
                "energy_consumption_kwh": energy_consumption,
                "measurements": self.measurements,
                "trees_equivalent": trees_equivalent,
                "mitigation_strategies": energy_solutions
            }
        except Exception as e:
            logger.error(f"Failed to generate emissions report: {str(e)}")
            return {"error": str(e)}


# Google Environmental Assessment and Reporting (EAR) API Integration
class GoogleEARIntegration:
    """Class for integrating with Google Environmental Assessment and Reporting API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.api_base_url = "https://ear.googleapis.com/v1"  # Example URL, would need to be replaced with actual URL
        self.websocket_url = "wss://ear-websocket.googleapis.com/v1"  # Example WebSocket URL
    
    def initialize(self, api_key=None):
        """Initialize the API integration"""
        if api_key:
            self.api_key = api_key
        
        try:
            # Test connection - This is a placeholder
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(f"{self.api_base_url}/status", headers=headers)
            
            if response.status_code == 200:
                logger.info("Google EAR API connection successful")
                return True
            else:
                logger.error(f"Google EAR API connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Google EAR API: {str(e)}")
            return False
    
    def assess_model(self, model_details):
        """Assess the environmental impact of an AI model"""
        try:
            import requests
            import json
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": {
                    "name": model_details.get("name", "Unknown Model"),
                    "version": model_details.get("version", "1.0"),
                    "parameters": model_details.get("parameters", 0),
                    "hardware": model_details.get("hardware", "CPU"),
                    "training_hours": model_details.get("training_hours", 0),
                    "inference_per_day": model_details.get("inference_per_day", 0)
                }
            }
            
            response = requests.post(
                f"{self.api_base_url}/models:assess",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                logger.info(f"Environmental assessment completed for {model_details.get('name', 'Unknown Model')}")
                return response.json()
            else:
                logger.error(f"Environmental assessment failed: {response.status_code}")
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to assess model: {str(e)}")
            return {"error": str(e)}
    
    def get_recommendations(self, assessment_id):
        """Get eco-friendly recommendations based on an assessment"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_base_url}/assessments/{assessment_id}/recommendations",
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"Retrieved recommendations for assessment {assessment_id}")
                return response.json()
            else:
                logger.error(f"Failed to get recommendations: {response.status_code}")
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get recommendations: {str(e)}")
            return {"error": str(e)}
    
    def open_websocket(self, callback):
        """Open a WebSocket connection for real-time environmental monitoring"""
        try:
            import websocket
            import threading
            import json
            
            def on_message(ws, message):
                data = json.loads(message)
                callback(data)
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                # Send authentication message
                ws.send(json.dumps({"auth": {"api_key": self.api_key}}))
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                self.websocket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection in a separate thread
            threading.Thread(target=ws.run_forever).start()
            
            return ws
        except Exception as e:
            logger.error(f"Failed to open WebSocket: {str(e)}")
            return None

# Page renderers
def render_dashboard():
    """Render the dashboard page safely"""
    try:
        st.markdown("""
        <h2>Dashboard</h2>
        <p>Overview of your AI security testing environment</p>
        """, unsafe_allow_html=True)
        
        # Quick stats in a row of cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(metric_card("Targets", len(st.session_state.targets), "Configured AI models"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(metric_card("Test Vectors", "9", "Available security tests"), unsafe_allow_html=True)
        
        with col3:
            vuln_count = len(st.session_state.test_results.get("vulnerabilities", [])) if st.session_state.test_results else 0
            st.markdown(metric_card("Vulnerabilities", vuln_count, "Identified issues"), unsafe_allow_html=True)
        
        with col4:
            risk_score = st.session_state.test_results.get("summary", {}).get("risk_score", 0) if st.session_state.test_results else 0
            st.markdown(metric_card("Risk Score", risk_score, "Overall security risk"), unsafe_allow_html=True)
        
        # Recent activity and status
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3>Recent Activity</h3>", unsafe_allow_html=True)
            
            if not st.session_state.test_results:
                st.markdown(card("No Recent Activity", "Run your first assessment to generate results.", "warning"), unsafe_allow_html=True)
            else:
                # Show the most recent vulnerabilities
                vulnerabilities = st.session_state.test_results.get("vulnerabilities", [])
                if vulnerabilities:
                    for vuln in vulnerabilities[:3]:  # Show top 3
                        severity_color = {
                            "low": get_theme()["text"],
                            "medium": get_theme()["warning"],
                            "high": get_theme()["warning"],
                            "critical": get_theme()["error"]
                        }.get(vuln["severity"], get_theme()["text"])
                        
                        st.markdown(f"""
                        <div class="card hover-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div class="card-title">{vuln["id"]}: {vuln["test_name"]}</div>
                                <div style="color: {severity_color}; font-weight: bold; text-transform: uppercase; font-size: 12px;">
                                    {vuln["severity"]}
                                </div>
                            </div>
                            <p>{vuln["details"]}</p>
                            <div style="font-size: 12px; opacity: 0.7;">Found in: {vuln["timestamp"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3>System Status</h3>", unsafe_allow_html=True)
            
            if st.session_state.running_test:
                st.markdown(card("Test in Progress", f"""
                <div style="margin-bottom: 10px;">
                    <div style="margin-bottom: 5px;">Progress:</div>
                    <div style="height: 10px; background-color: rgba(255,255,255,0.1); border-radius: 5px;">
                        <div style="height: 10px; width: {st.session_state.progress*100}%; background-color: {get_theme()["primary"]}; border-radius: 5px;"></div>
                    </div>
                    <div style="text-align: right; font-size: 12px; margin-top: 5px;">{int(st.session_state.progress*100)}%</div>
                </div>
                <div>Vulnerabilities found: {st.session_state.vulnerabilities_found}</div>
                """, "warning"), unsafe_allow_html=True)
            else:
                st.markdown(card("System Ready", """
                <p>All systems operational and ready to run assessments.</p>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background-color: #4CAF50; border-radius: 50%; margin-right: 5px;"></div>
                    <div>API Connection: Active</div>
                </div>
                """, "success"), unsafe_allow_html=True)
        
        # Test vector overview
        st.markdown("<h3>Test Vector Overview</h3>", unsafe_allow_html=True)
        
        # Create a radar chart for test coverage
        try:
            test_vectors = get_mock_test_vectors()
            categories = list(set(tv["category"] for tv in test_vectors))
            
            # Count test vectors by category
            category_counts = {}
            for cat in categories:
                category_counts[cat] = sum(1 for tv in test_vectors if tv["category"] == cat)
            
            # Create the data for the radar chart
            fig = go.Figure()
            
            primary_color = get_theme()["primary"]
            r_value = int(primary_color[1:3], 16) if len(primary_color) >= 7 else 29
            g_value = int(primary_color[3:5], 16) if len(primary_color) >= 7 else 185
            b_value = int(primary_color[5:7], 16) if len(primary_color) >= 7 else 84
            
            fig.add_trace(go.Scatterpolar(
                r=list(category_counts.values()),
                theta=list(category_counts.keys()),
                fill='toself',
                fillcolor=f'rgba({r_value}, {g_value}, {b_value}, 0.3)',
                line=dict(color=primary_color),
                name='Test Coverage'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(category_counts.values()) + 1]
                    )
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_theme()["text"])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering radar chart: {str(e)}")
            st.error("Failed to render radar chart")
        
        # Quick actions with Streamlit buttons
        st.markdown("<h3>Quick Actions</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add New Target", use_container_width=True, key="dashboard_add_target"):
                set_page("Target Management")
                safe_rerun()
        
        with col2:
            if st.button("🧪 Run Assessment", use_container_width=True, key="dashboard_run_assessment"):
                set_page("Run Assessment")
                safe_rerun()
        
        with col3:
            if st.button("📊 View Results", use_container_width=True, key="dashboard_view_results"):
                set_page("Results Analyzer")
                safe_rerun()
                
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error rendering dashboard: {str(e)}")

def render_target_management():
    """Render the target management page safely"""
    try:
        st.markdown("""
        <h2>Target Management</h2>
        <p>Add and configure AI models to test</p>
        """, unsafe_allow_html=True)
        
        # Show existing targets
        if st.session_state.targets:
            st.markdown("<h3>Your Targets</h3>", unsafe_allow_html=True)
            
            # Use columns for better layout
            cols = st.columns(3)
            for i, target in enumerate(st.session_state.targets):
                col = cols[i % 3]
                with col:
                    with st.container():
                        st.markdown(f"### {target['name']}")
                        st.markdown(f"**Endpoint:** {target['endpoint']}")
                        st.markdown(f"**Type:** {target.get('type', 'Unknown')}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✏️ Edit", key=f"edit_target_{i}", use_container_width=True):
                                # In a real app, this would open an edit dialog
                                st.info("Edit functionality would open here")
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_target_{i}", use_container_width=True):
                                # Remove the target
                                st.session_state.targets.pop(i)
                                st.success(f"Target '{target['name']}' deleted")
                                safe_rerun()
        
        # Add new target form
        st.markdown("<h3>Add New Target</h3>", unsafe_allow_html=True)
        
        with st.form("add_target_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_name = st.text_input("Target Name")
                target_endpoint = st.text_input("API Endpoint URL")
                target_type = st.selectbox("Model Type", ["LLM", "Content Filter", "Embedding", "Classification", "Other"])
            
            with col2:
                api_key = st.text_input("API Key", type="password")
                target_description = st.text_area("Description")
            
            submit_button = st.form_submit_button("Add Target")
            
            if submit_button:
                try:
                    if not target_name or not target_endpoint:
                        st.error("Name and endpoint are required")
                    else:
                        new_target = {
                            "name": target_name,
                            "endpoint": target_endpoint,
                            "type": target_type,
                            "api_key": api_key,
                            "description": target_description
                        }
                        st.session_state.targets.append(new_target)
                        st.success(f"Target '{target_name}' added successfully!")
                        logger.info(f"Added new target: {target_name}")
                        safe_rerun()
                except Exception as e:
                    logger.error(f"Error adding target: {str(e)}")
                    st.error(f"Failed to add target: {str(e)}")
        
        # Import/Export
        st.markdown("<h3>Import/Export Targets</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Import Targets", type=["json"], key="target_import")
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read()
                    imported_targets = json.loads(content)
                    
                    if isinstance(imported_targets, list):
                        # Validate the imported targets
                        valid_targets = []
                        for target in imported_targets:
                            if isinstance(target, dict) and "name" in target and "endpoint" in target:
                                valid_targets.append(target)
                        
                        if valid_targets:
                            st.session_state.targets.extend(valid_targets)
                            st.success(f"Successfully imported {len(valid_targets)} targets")
                            logger.info(f"Imported {len(valid_targets)} targets")
                            safe_rerun()
                        else:
                            st.error("No valid targets found in the imported file")
                    else:
                        st.error("Invalid JSON format. Expected a list of targets.")
                except Exception as e:
                    logger.error(f"Error importing targets: {str(e)}")
                    st.error(f"Failed to import targets: {str(e)}")
        
        with col2:
            if st.session_state.targets:
                try:
                    targets_json = json.dumps(st.session_state.targets, indent=2)
                    st.download_button(
                        label="Export Targets",
                        data=targets_json,
                        file_name=f"targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="target_export"
                    )
                except Exception as e:
                    logger.error(f"Error exporting targets: {str(e)}")
                    st.error(f"Failed to export targets: {str(e)}")
            else:
                st.button("Export Targets", disabled=True, key="export_disabled")
    
    except Exception as e:
        logger.error(f"Error rendering target management: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in target management: {str(e)}")

def render_test_configuration():
    """Render the test configuration page safely"""
    try:
        st.markdown("""
        <h2>Test Configuration</h2>
        <p>Customize your security assessment</p>
        """, unsafe_allow_html=True)
        
        # Test vector selection
        test_vectors = get_mock_test_vectors()
        
        # Group by category
        categories = {}
        for tv in test_vectors:
            if tv["category"] not in categories:
                categories[tv["category"]] = []
            categories[tv["category"]].append(tv)
        
        # Create tabs for each category
        try:
            tabs = st.tabs(list(categories.keys()))
            
            for i, (category, tab) in enumerate(zip(categories.keys(), tabs)):
                with tab:
                    st.markdown(f"<h3>{category.upper()} Test Vectors</h3>", unsafe_allow_html=True)
                    
                    # Create a list of test vectors
                    for j, tv in enumerate(categories[category]):
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"### {tv['name']}")
                                st.markdown(f"**Severity:** {tv['severity'].upper()}")
                                st.markdown(f"**Category:** {tv['category'].upper()}")
                            
                            with col2:
                                # Use a checkbox to enable/disable
                                is_enabled = st.checkbox("Enable", value=True, key=f"enable_{tv['id']}")
        except Exception as e:
            logger.error(f"Error rendering test vector tabs: {str(e)}")
            st.error(f"Failed to render test vectors: {str(e)}")
            
            # Fallback: Show test vectors in a simple list
            st.markdown("### Test Vectors")
            for tv in test_vectors:
                st.markdown(f"- **{tv['name']}** ({tv['category']}, {tv['severity']})")
        
        # Advanced configuration
        st.markdown("<h3>Advanced Configuration</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                test_duration = st.slider("Maximum Test Duration (minutes)", 5, 120, 30, key="test_duration")
                test_variations = st.number_input("Test Variations per Vector", 1, 1000, 10, key="test_variations")
                concurrency = st.slider("Concurrency Level", 1, 16, 4, key="concurrency")
            
            with col2:
                test_profile = st.selectbox("Test Profile", ["Standard", "Thorough", "Extreme", "Custom"], key="test_profile")
                focus_area = st.radio("Focus Area", ["General Security", "AI Safety", "Compliance", "All"], key="focus_area")
                save_detailed = st.checkbox("Save Detailed Results", value=True, key="save_detailed")
        except Exception as e:
            logger.error(f"Error rendering advanced configuration: {str(e)}")
            st.error(f"Failed to render advanced configuration: {str(e)}")
        
        # Save configuration button
        if st.button("Save Configuration", key="save_test_config"):
            st.success("Test configuration saved successfully!")
            logger.info("Test configuration saved")
        
        # Show configuration summary
        st.markdown("<h3>Configuration Summary</h3>", unsafe_allow_html=True)
        
        try:
            # Count enabled test vectors
            enabled_count = sum(1 for tv in test_vectors if st.session_state.get(f"enable_{tv['id']}", True))
            
            st.markdown(card("Test Parameters", f"""
            <ul>
                <li><strong>Enabled Test Vectors:</strong> {enabled_count} of {len(test_vectors)}</li>
                <li><strong>Estimated Duration:</strong> {test_duration} minutes</li>
                <li><strong>Total Test Cases:</strong> {enabled_count * test_variations} ({enabled_count} vectors × {test_variations} variations)</li>
                <li><strong>Profile:</strong> {test_profile}</li>
                <li><strong>Focus Area:</strong> {focus_area}</li>
            </ul>
            """), unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering configuration summary: {str(e)}")
            st.error(f"Failed to render configuration summary: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering test configuration: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in test configuration: {str(e)}")

def render_run_assessment():
    """Render the run assessment page safely"""
    try:
        st.markdown("""
        <h2>Run Assessment</h2>
        <p>Execute security tests against your targets</p>
        """, unsafe_allow_html=True)
        
        # Check if targets exist
        if not st.session_state.targets:
            st.warning("No targets configured. Please add a target first.")
            if st.button("Add Target", key="run_add_target"):
                set_page("Target Management")
                safe_rerun()
            return
        
        # Check if a test is already running
        if st.session_state.running_test:
            # Show progress
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                progress_bar = st.progress(st.session_state.progress)
                st.markdown(f"**Progress:** {int(st.session_state.progress*100)}%")
                st.markdown(f"**Vulnerabilities found:** {st.session_state.vulnerabilities_found}")
            
            # Stop button
            if st.button("Stop Test", key="stop_test"):
                st.session_state.running_test = False
                logger.info("Test stopped by user")
                st.warning("Test stopped by user")
                safe_rerun()
        else:
            # Test configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Select Target</h3>", unsafe_allow_html=True)
                target_options = [t["name"] for t in st.session_state.targets]
                selected_target = st.selectbox("Target", target_options, key="run_target")
            
            with col2:
                st.markdown("<h3>Test Parameters</h3>", unsafe_allow_html=True)
                test_duration = st.slider("Test Duration (seconds)", 5, 60, 30, key="run_duration", 
                                         help="For demonstration purposes, we're using seconds. In a real system, this would be minutes.")
            
            # Get test vectors
            test_vectors = get_mock_test_vectors()
            
            # Show test vector selection
            st.markdown("<h3>Select Test Vectors</h3>", unsafe_allow_html=True)
            
            # Group by category
            categories = {}
            for tv in test_vectors:
                if tv["category"] not in categories:
                    categories[tv["category"]] = []
                categories[tv["category"]].append(tv)
            
            # Create columns for each category
            try:
                cols = st.columns(len(categories))
                
                selected_vectors = []
                for i, (category, col) in enumerate(zip(categories.keys(), cols)):
                    with col:
                        st.markdown(f"<div style='text-align: center; text-transform: uppercase; font-weight: bold; margin-bottom: 10px;'>{category}</div>", unsafe_allow_html=True)
                        
                        for tv in categories[category]:
                            if st.checkbox(tv["name"], value=True, key=f"run_tv_{tv['id']}"):
                                selected_vectors.append(tv)
            except Exception as e:
                logger.error(f"Error rendering test vector selection: {str(e)}")
                st.error(f"Failed to render test vector selection: {str(e)}")
                
                # Fallback: Use multiselect
                st.markdown("### Select Test Vectors")
                vector_names = [tv["name"] for tv in test_vectors]
                selected_names = st.multiselect("Test Vectors", vector_names, default=vector_names, key="fallback_vectors")
                selected_vectors = [tv for tv in test_vectors if tv["name"] in selected_names]
            
            # Environmental impact tracking option
            st.markdown("<h3>Environmental Impact Tracking</h3>", unsafe_allow_html=True)
            track_carbon = st.checkbox("Track Carbon Emissions During Test", value=True, key="track_carbon_emissions")
            
            if track_carbon:
                st.info("Carbon tracking will be enabled during the test to measure environmental impact")
            
            # Run test button
            if st.button("Run Assessment", use_container_width=True, type="primary", key="start_assessment"):
                try:
                    if not selected_vectors:
                        st.error("Please select at least one test vector")
                    else:
                        # Find the selected target object
                        target = next((t for t in st.session_state.targets if t["name"] == selected_target), None)
                        
                        if target:
                            # Initialize carbon tracking if requested
                            if track_carbon and 'carbon_tracker' not in st.session_state:
                                st.session_state.carbon_tracker = CarbonImpactTracker()
                                st.session_state.carbon_tracker.initialize_tracker(f"Security Test - {target['name']}")
                            
                            if track_carbon:
                                st.session_state.carbon_tracker.start_tracking()
                                st.session_state.carbon_tracking_active = True
                            
                            # Start the test in a background thread
                            test_thread = threading.Thread(
                                target=run_mock_test,
                                args=(target, selected_vectors, test_duration)
                            )
                            test_thread.daemon = True
                            test_thread.start()
                            
                            # Track the thread
                            st.session_state.active_threads.append(test_thread)
                            
                            st.session_state.running_test = True
                            logger.info(f"Started test against {target['name']} with {len(selected_vectors)} vectors")
                            st.success("Test started!")
                            safe_rerun()
                        else:
                            st.error("Selected target not found")
                except Exception as e:
                    logger.error(f"Error starting test: {str(e)}")
                    st.error(f"Failed to start test: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering run assessment: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in run assessment: {str(e)}")

def render_results_analyzer():
    """Render the results analyzer page safely"""
    try:
        st.markdown("""
        <h2>Results Analyzer</h2>
        <p>Explore and analyze security assessment results</p>
        """, unsafe_allow_html=True)
        
        # Check if there are results to display
        if not st.session_state.test_results:
            st.warning("No Results Available - Run an assessment to generate results.")
            
            if st.button("Go to Run Assessment", key="results_goto_run"):
                set_page("Run Assessment")
                safe_rerun()
            return
        
        # Results summary
        results = st.session_state.test_results
        
        # Check if results contains an error
        if results.get("error", False):
            st.error(f"The last test resulted in an error: {results.get('error_message', 'Unknown error')}")
            if st.button("Clear Error and Run New Test", key="clear_error"):
                st.session_state.test_results = {}
                set_page("Run Assessment")
                safe_rerun()
            return
        
        vulnerabilities = results.get("vulnerabilities", [])
        summary = results.get("summary", {})
        
        # Create header with summary metrics
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3>Assessment Results: {results.get("target", "Unknown Target")}</h3>
            <div style="opacity: 0.7;">Completed: {results.get("timestamp", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tests Run", summary.get("total_tests", 0))
        
        with col2:
            st.metric("Vulnerabilities", summary.get("vulnerabilities_found", 0))
        
        with col3:
            st.metric("Risk Score", summary.get("risk_score", 0))
        
        # Visualizations
        st.markdown("<h3>Vulnerability Overview</h3>", unsafe_allow_html=True)
        
        # Prepare data for charts
        if vulnerabilities:
            try:
                # Count vulnerabilities by severity
                severity_counts = {}
                for vuln in vulnerabilities:
                    severity = vuln.get("severity", "unknown")
                    if severity not in severity_counts:
                        severity_counts[severity] = 0
                    severity_counts[severity] += 1
                
                # Count vulnerabilities by test vector
                vector_counts = {}
                for vuln in vulnerabilities:
                    vector = vuln.get("test_name", "unknown")
                    if vector not in vector_counts:
                        vector_counts[vector] = 0
                    vector_counts[vector] += 1
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create pie chart for severity distribution
                    labels = list(severity_counts.keys())
                    values = list(severity_counts.values())
                    
                    colors = {
                        "low": "green",
                        "medium": "yellow",
                        "high": "orange",
                        "critical": "red",
                        "unknown": "gray"
                    }
                    
                    fig = px.pie(
                        names=labels,
                        values=values,
                        title="Vulnerabilities by Severity",
                        color=labels,
                        color_discrete_map={label: colors.get(label, "gray") for label in labels}
                    )
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=get_theme()["text"])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create bar chart for test vector distribution
                    fig = px.bar(
                        x=list(vector_counts.keys()),
                        y=list(vector_counts.values()),
                        title="Vulnerabilities by Test Vector",
                        labels={"x": "Test Vector", "y": "Vulnerabilities"},
                        color_discrete_sequence=[get_theme()["primary"]]
                    )
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=get_theme()["text"])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering charts: {str(e)}")
                st.error(f"Failed to render charts: {str(e)}")
        
        # Detailed vulnerability listing
        st.markdown("<h3>Detailed Findings</h3>", unsafe_allow_html=True)
        
        if vulnerabilities:
            try:
                # Create tabs for different severity levels
                severities = list(set(vuln["severity"] for vuln in vulnerabilities if "severity" in vuln))
                severities.sort(key=lambda s: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(s, 4))
                
                # Add "All" tab at the beginning
                tabs = st.tabs(["All"] + severities)
                
                with tabs[0]:  # "All" tab
                    for i, vuln in enumerate(vulnerabilities):
                        severity = vuln.get("severity", "unknown")
                        severity_emoji = {
                            "low": "🟢",
                            "medium": "🟡",
                            "high": "🟠",
                            "critical": "🔴",
                            "unknown": "⚪"
                        }.get(severity, "⚪")
                        
                        with st.expander(f"{severity_emoji} {vuln.get('id', 'Unknown')}: {vuln.get('test_name', 'Unknown Test')}"):
                            st.markdown(f"**Severity:** {severity.upper()}")
                            st.markdown(f"**Details:** {vuln.get('details', 'No details available.')}")
                            st.markdown(f"**Found:** {vuln.get('timestamp', 'Unknown')}")
                
                # Create content for each severity tab
                for i, severity in enumerate(severities):
                    with tabs[i+1]:  # +1 because "All" is the first tab
                        severity_vulns = [v for v in vulnerabilities if v.get("severity") == severity]
                        
                        for j, vuln in enumerate(severity_vulns):
                            severity_emoji = {
                                "low": "🟢",
                                "medium": "🟡",
                                "high": "🟠",
                                "critical": "🔴",
                                "unknown": "⚪"
                            }.get(severity, "⚪")
                            
                            with st.expander(f"{severity_emoji} {vuln.get('id', 'Unknown')}: {vuln.get('test_name', 'Unknown Test')}"):
                                st.markdown(f"**Severity:** {severity.upper()}")
                                st.markdown(f"**Details:** {vuln.get('details', 'No details available.')}")
                                st.markdown(f"**Found:** {vuln.get('timestamp', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error rendering vulnerability details: {str(e)}")
                st.error(f"Failed to render vulnerability details: {str(e)}")
                
                # Fallback: Simple list of vulnerabilities
                for vuln in vulnerabilities:
                    st.markdown(f"- **{vuln.get('id', 'Unknown')}**: {vuln.get('details', 'No details')}")
        else:
            st.info("No vulnerabilities were found in this assessment.")
        
        # Export results
        st.markdown("<h3>Export Results</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(results, indent=2),
                    file_name=f"security_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
            except Exception as e:
                logger.error(f"Error preparing JSON download: {str(e)}")
                st.error(f"Failed to prepare JSON download: {str(e)}")
        
        with col2:
            try:
                if vulnerabilities:
                    # Convert vulnerabilities to DataFrame
                    df = pd.DataFrame(vulnerabilities)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV Vulnerabilities",
                        data=csv,
                        file_name=f"vulnerabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                else:
                    st.button("Download CSV Vulnerabilities", disabled=True, key="download_csv_disabled")
            except Exception as e:
                logger.error(f"Error preparing CSV download: {str(e)}")
                st.error(f"Failed to prepare CSV download: {str(e)}")
        
        with col3:
            # Add Excel export option
            try:
                if vulnerabilities:
                    import io
                    import pandas as pd
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    
                    # Create a Pandas Excel writer
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Convert vulnerabilities to DataFrame
                        df = pd.DataFrame(vulnerabilities)
                        df.to_excel(writer, sheet_name='Vulnerabilities', index=False)
                        
                        # Create a summary sheet
                        summary_data = {
                            'Metric': ['Tests Run', 'Vulnerabilities Found', 'Risk Score', 'Date'],
                            'Value': [
                                summary.get('total_tests', 0),
                                summary.get('vulnerabilities_found', 0),
                                summary.get('risk_score', 0),
                                results.get('timestamp', 'Unknown')
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Get the value from the BytesIO buffer
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"security_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"
                    )
                else:
                    st.button("Download Excel Report", disabled=True, key="download_excel_disabled")
            except Exception as e:
                logger.error(f"Error preparing Excel download: {str(e)}")
                st.error(f"Failed to prepare Excel download: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering results analyzer: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in results analyzer: {str(e)}")

def render_ethical_ai_testing():
    """Render the ethical AI testing page safely"""
    try:
        st.markdown("""
        <h2>Ethical AI Testing</h2>
        <p>Comprehensive assessment of AI systems against OWASP, NIST, and ethical guidelines</p>
        """, unsafe_allow_html=True)
        
        # Check if targets exist
        if not st.session_state.targets:
            st.warning("No targets configured. Please add a target first.")
            if st.button("Add Target", key="ethical_add_target"):
                set_page("Target Management")
                safe_rerun()
            return
        
        # Create tabs for different testing frameworks
        try:
            tabs = st.tabs(["OWASP LLM", "NIST Framework", "Fairness & Bias", "Privacy Compliance", "Synthetic Extreme"])
            
            with tabs[0]:
                st.markdown("<h3>OWASP LLM Top 10 Testing</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                This module tests AI systems against the OWASP Top 10 for Large Language Model Applications:
                
                - Prompt Injection
                - Insecure Output Handling
                - Training Data Poisoning
                - Model Denial of Service
                - Supply Chain Vulnerabilities
                - Sensitive Information Disclosure
                - Insecure Plugin Design
                - Excessive Agency
                - Overreliance
                - Model Theft
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [t["name"] for t in st.session_state.targets]
                    st.selectbox("Select Target", target_options, key="owasp_target")
                
                with col2:
                    st.multiselect("Select Tests", [
                        "Prompt Injection",
                        "Insecure Output Handling",
                        "Sensitive Information Disclosure",
                        "Excessive Agency"
                    ], default=["Prompt Injection", "Insecure Output Handling"], key="owasp_tests")
                
                if st.button("Run OWASP LLM Tests", key="run_owasp"):
                    st.info("OWASP LLM testing would start here")
                    # In a real implementation, this would start the tests
            
            with tabs[1]:
                st.markdown("<h3>NIST AI Risk Management Framework</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                This module evaluates AI systems against the NIST AI Risk Management Framework:
                
                - Governance
                - Mapping
                - Measurement
                - Management
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [t["name"] for t in st.session_state.targets]
                    st.selectbox("Select Target", target_options, key="nist_target")
                
                with col2:
                    st.multiselect("Select Framework Components", [
                        "Governance",
                        "Mapping",
                        "Measurement",
                        "Management"
                    ], default=["Governance", "Management"], key="nist_components")
                
                if st.button("Run NIST Framework Assessment", key="run_nist"):
                    st.info("NIST Framework assessment would start here")
            
            with tabs[2]:
                st.markdown("<h3>Fairness & Bias Testing</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                This module tests AI systems for fairness and bias issues:
                
                - Demographic Parity
                - Equal Opportunity
                - Disparate Impact
                - Representation Bias
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [t["name"] for t in st.session_state.targets]
                    st.selectbox("Select Target", target_options, key="fairness_target")
                
                with col2:
                    st.multiselect("Select Fairness Metrics", [
                        "Demographic Parity",
                        "Equal Opportunity",
                        "Disparate Impact",
                        "Representation Bias"
                    ], default=["Demographic Parity"], key="fairness_metrics")
                
                st.text_area("Demographic Groups (one per line)", "Group A\nGroup B\nGroup C\nGroup D", key="demographic_groups")
                
                if st.button("Run Fairness Assessment", key="run_fairness"):
                    st.info("Fairness assessment would start here")
                    # Link to our dedicated bias testing page
                    st.markdown("For more comprehensive bias testing, visit our [Bias Testing](#) page")
                    if st.button("Go to Bias Testing", key="goto_bias_testing"):
                        set_page("Bias Testing")
                        safe_rerun()
            
            with tabs[3]:
                st.markdown("<h3>Privacy Compliance Testing</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                This module tests AI systems for compliance with privacy regulations:
                
                - GDPR
                - CCPA
                - HIPAA
                - PIPEDA
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [t["name"] for t in st.session_state.targets]
                    st.selectbox("Select Target", target_options, key="privacy_target")
                
                with col2:
                    st.multiselect("Select Regulations", [
                        "GDPR",
                        "CCPA",
                        "HIPAA",
                        "PIPEDA"
                    ], default=["GDPR"], key="privacy_regulations")
                
                if st.button("Run Privacy Assessment", key="run_privacy"):
                    st.info("Privacy assessment would start here")
            
            with tabs[4]:
                st.markdown("<h3>Synthetic Extreme Testing</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                This module performs rigorous synthetic testing focusing on AI-specific vulnerabilities:
                
                - Jailbreaking
                - Advanced Prompt Injection
                - Data Extraction
                - Boundary Testing
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [t["name"] for t in st.session_state.targets]
                    st.selectbox("Select Target", target_options, key="extreme_target")
                
                with col2:
                    st.multiselect("Select Techniques", [
                        "Jailbreaking",
                        "Advanced Prompt Injection",
                        "Data Extraction",
                        "Boundary Testing"
                    ], default=["Jailbreaking"], key="extreme_techniques")
                
                st.slider("Testing Intensity", 1, 10, 5, key="testing_intensity")
                
                # Add environmental impact option
                impact_tracking = st.checkbox("Monitor Environmental Impact", value=True, key="monitor_env_impact")
                if impact_tracking:
                    st.info("Environmental impact will be monitored and reported during testing")
                
                if st.button("Run Extreme Testing", key="run_extreme"):
                    st.info("Synthetic extreme testing would start here")
        
        except Exception as e:
            logger.error(f"Error rendering ethical AI tabs: {str(e)}")
            st.error(f"Failed to render ethical AI testing interface: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering ethical AI testing: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in ethical AI testing: {str(e)}")

def render_high_volume_testing():
    """Render the high-volume testing page safely"""
    try:
        st.markdown("""
        <h2>High-Volume Testing</h2>
        <p>Autonomous, high-throughput testing for AI systems</p>
        """, unsafe_allow_html=True)
        
        # Check if targets exist
        if not st.session_state.targets:
            st.warning("No targets configured. Please add a target first.")
            if st.button("Add Target", key="highvol_add_target"):
                set_page("Target Management")
                safe_rerun()
            return
        
        # Configuration section
        st.markdown("<h3>Testing Configuration</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                target_options = [t["name"] for t in st.session_state.targets]
                st.selectbox("Select Target", target_options, key="highvol_target")
                
                total_tests = st.slider("Total Tests (thousands)", 10, 1000, 100, key="highvol_tests")
                
                max_runtime = st.number_input("Max Runtime (hours)", 1, 24, 3, key="highvol_runtime")
            
            with col2:
                st.multiselect("Test Vectors", [
                    "Prompt Injection",
                    "Jailbreaking",
                    "Data Extraction",
                    "Input Manipulation",
                    "Boundary Testing"
                ], default=["Prompt Injection", "Jailbreaking"], key="highvol_vectors")
                
                parallelism = st.selectbox("Parallelism", ["Low (4 workers)", "Medium (8 workers)", "High (16 workers)", "Extreme (32 workers)"], key="highvol_parallel")
                
                save_only_vulns = st.checkbox("Save Only Vulnerabilities", value=True, key="highvol_save_vulns")
        except Exception as e:
            logger.error(f"Error rendering high-volume configuration: {str(e)}")
            st.error(f"Failed to render high-volume testing configuration: {str(e)}")
        
        # Environment monitoring section
        st.markdown("<h3>Environmental Monitoring</h3>", unsafe_allow_html=True)
        
        try:
            carbon_aware = st.checkbox("Enable Carbon-Aware Scheduling", value=True, key="carbon_aware_scheduling",
                                       help="Adjust testing intensity based on carbon intensity of electricity grid")
            
            if carbon_aware:
                st.success("Carbon-aware scheduling will prioritize testing during low-carbon periods")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    carbon_threshold = st.slider("Carbon Intensity Threshold (gCO2/kWh)", 100, 500, 300, key="carbon_threshold",
                                                help="Testing will slow down when carbon intensity exceeds this threshold")
                
                with col2:
                    st.selectbox("Grid Region", [
                        "us-west",
                        "us-east",
                        "europe-west",
                        "europe-north",
                        "asia-east",
                        "asia-southeast"
                    ], key="grid_region")
        except Exception as e:
            logger.error(f"Error rendering environmental monitoring: {str(e)}")
            st.error(f"Failed to render environmental monitoring: {str(e)}")
        
        # Resource monitoring
        st.markdown("<h3>Resource Monitoring</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2, col3 = st.columns(3)
            
            worker_count = {"Low (4 workers)": 4, "Medium (8 workers)": 8, "High (16 workers)": 16, "Extreme (32 workers)": 32}
            selected_workers = worker_count.get(st.session_state.get("highvol_parallel", "Medium (8 workers)"), 8)
            
            with col1:
                st.metric("Max Workers", selected_workers)
            
            with col2:
                st.metric("Rate Limit", "100 req/sec")
            
            with col3:
                st.metric("Memory Limit", "8 GB")
        except Exception as e:
            logger.error(f"Error rendering resource monitoring: {str(e)}")
            st.error(f"Failed to render resource monitoring: {str(e)}")
        
        # Start testing button
        if st.button("Start High-Volume Testing", type="primary", use_container_width=True, key="start_highvol"):
            try:
                st.success("High-volume testing started! This would typically run for several hours in a production environment.")
                
                # Create placeholders for progress updates
                progress_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Simulate progress updates
                for i in range(101):
                    # Check if the page has been navigated away from
                    if st.session_state.current_page != "High-Volume Testing":
                        break
                    
                    with progress_placeholder:
                        st.progress(i / 100)
                    
                    # Update metrics every 10%
                    if i % 10 == 0:
                        with metrics_placeholder:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Tests Completed", f"{i * 1000:,}")
                            with col2:
                                vulnerabilities = int(i * 1000 * 0.02)  # 2% find rate
                                st.metric("Vulnerabilities", f"{vulnerabilities:,}")
                            with col3:
                                st.metric("Tests/Second", f"{random.randint(80, 120):,}")
                            with col4:
                                # Show CO2 emissions
                                co2 = i * 0.005  # Simulated CO2 emissions in kg
                                st.metric("CO2 Emissions", f"{co2:.2f} kg")
                    
                    time.sleep(0.05)  # Just for demonstration
                
                # Remove progress indicators
                progress_placeholder.empty()
                
                # Show completion message
                st.success("Testing completed! 100,000 tests executed, 2,000 vulnerabilities identified.")
                
                # Sample results visualization
                st.markdown("<h3>Results Overview</h3>", unsafe_allow_html=True)
                
                # Generate some mock data
                vector_names = ["Prompt Injection", "Jailbreaking", "Data Extraction", "Input Manipulation", "Boundary Testing"]
                vulnerability_counts = [random.randint(200, 600) for _ in range(5)]
                
                # Create bar chart
                fig = px.bar(
                    x=vector_names,
                    y=vulnerability_counts,
                    labels={"x": "Test Vector", "y": "Vulnerabilities Found"},
                    color=vulnerability_counts,
                    color_continuous_scale="Viridis"
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=get_theme()["text"])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Environmental impact section
                st.markdown("<h3>Environmental Impact</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Carbon Emissions", "2.46 kg CO2eq")
                
                with col2:
                    st.metric("Energy Consumed", "4.9 kWh")
                
                with col3:
                    st.metric("Carbon Intensity", "502 gCO2/kWh")
                
                st.info("💡 Suggestion: Running this test during off-peak hours could reduce emissions by up to 35%")
 st.info("💡 Suggestion: Running this test during off-peak hours could reduce emissions by up to 35%")
except Exception as e:
    logger.error(f"Error displaying energy optimization suggestion: {str(e)}")
    st.error("Error displaying energy optimization information")
                
                # Optimization recommendations
                st.markdown("<h4>Optimization Recommendations</h4>", unsafe_allow_html=True)
                
                recommendations = [
                    {
                        "title": "Reduce Test Frequency",
                        "description": "High-volume testing once per week instead of daily could reduce emissions by 80%",
                        "difficulty": "Easy"
                    },
                    {
                        "title": "Carbon-Aware Scheduling",
                        "description": "Schedule tests during periods of low carbon intensity in your region",
                        "difficulty": "Medium"
                    },
                    {
                        "title": "Optimize Worker Count",
                        "description": "Reducing from 16 to 8 workers showed only 10% longer runtime but 40% less energy",
                        "difficulty": "Easy"
                    }
                ]
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"{i+1}. {rec['title']} ({rec['difficulty']})", expanded=i==0):
                        st.markdown(rec["description"])
            except Exception as e:
                logger.error(f"Error in high-volume testing simulation: {str(e)}")
                st.error(f"Error in high-volume testing: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering high-volume testing: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in high-volume testing: {str(e)}")

def render_settings():
    """Render the settings page safely"""
    try:
        st.markdown("""
        <h2>Settings</h2>
        <p>Configure application settings and preferences</p>
        """, unsafe_allow_html=True)
        
        # Theme settings
        st.markdown("<h3>Theme Settings</h3>", unsafe_allow_html=True)
        
        theme_option = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.current_theme == "dark" else 1, key="settings_theme")
        if theme_option == "Dark" and st.session_state.current_theme != "dark":
            st.session_state.current_theme = "dark"
            logger.info("Theme set to dark")
            safe_rerun()
        elif theme_option == "Light" and st.session_state.current_theme != "light":
            st.session_state.current_theme = "light"
            logger.info("Theme set to light")
            safe_rerun()
        
        # API settings
        st.markdown("<h3>API Settings</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                api_base_url = st.text_input("API Base URL", "https://api.example.com/v1", key="api_base_url")
            
            with col2:
                default_api_key = st.text_input("Default API Key", type="password", key="default_api_key")
            
            # Save API settings
            if st.button("Save API Settings", key="save_api"):
                st.success("API settings saved successfully!")
                logger.info("API settings updated")
        except Exception as e:
            logger.error(f"Error rendering API settings: {str(e)}")
            st.error(f"Failed to render API settings: {str(e)}")
        
        # Testing settings
        st.markdown("<h3>Testing Settings</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                default_duration = st.number_input("Default Test Duration (minutes)", 5, 120, 30, key="default_duration")
                request_timeout = st.number_input("Request Timeout (seconds)", 1, 60, 10, key="request_timeout")
            
            with col2:
                max_concurrent = st.number_input("Maximum Concurrent Tests", 1, 32, 4, key="max_concurrent_tests")
                save_logs = st.checkbox("Save Detailed Logs", value=True, key="save_detailed_logs")
            
            # Save testing settings
            if st.button("Save Testing Settings", key="save_testing"):
                st.success("Testing settings saved successfully!")
                logger.info("Testing settings updated")
        except Exception as e:
            logger.error(f"Error rendering testing settings: {str(e)}")
            st.error(f"Failed to render testing settings: {str(e)}")
        
        # Environment settings
        st.markdown("<h3>Environmental Settings</h3>", unsafe_allow_html=True)
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                carbon_tracking = st.checkbox("Enable Carbon Tracking", value=True, key="settings_carbon_tracking")
                carbon_api_key = st.text_input("Carbon API Key (optional)", type="password", key="carbon_api_key",
                                              help="API key for accessing external carbon intensity data")
            
            with col2:
                preferred_region = st.selectbox("Preferred Compute Region", [
                    "us-west",
                    "us-east",
                    "europe-west",
                    "europe-north",
                    "asia-east",
                    "asia-southeast"
                ], index=2, key="preferred_region",
                help="Region with lowest carbon intensity will be preferred for compute-intensive tasks")
                
                emissions_threshold = st.slider("Emissions Alert Threshold (kg CO2)", 0.0, 10.0, 1.0, key="emissions_threshold",
                                              help="Alert when test emissions exceed this threshold")
            
            # Save environmental settings
            if st.button("Save Environmental Settings", key="save_env_settings"):
                st.success("Environmental settings saved successfully!")
                logger.info("Environmental settings updated")
        except Exception as e:
            logger.error(f"Error rendering environmental settings: {str(e)}")
            st.error(f"Failed to render environmental settings: {str(e)}")
        
        # Notifications
        st.markdown("<h3>Notifications</h3>", unsafe_allow_html=True)
        
        try:
            email_notifications = st.checkbox("Email Notifications", value=False, key="email_notifications")
            
            if email_notifications:
                email_address = st.text_input("Email Address", key="notification_email")
                notification_events = st.multiselect("Notify On", ["Test Completion", "Critical Vulnerability", "Error", "Carbon Threshold Exceeded"], default=["Test Completion", "Critical Vulnerability"], key="notification_events")
            
            # Save notification settings
            if st.button("Save Notification Settings", key="save_notifications"):
                st.success("Notification settings saved successfully!")
                logger.info("Notification settings updated")
        except Exception as e:
            logger.error(f"Error rendering notification settings: {str(e)}")
            st.error(f"Failed to render notification settings: {str(e)}")
        
        # System information
        st.markdown("<h3>System Information</h3>", unsafe_allow_html=True)
        
        try:
            # Get system info
            import platform
            
            system_info = f"""
            - Python Version: {platform.python_version()}
            - Operating System: {platform.system()} {platform.release()}
            - Streamlit Version: {st.__version__}
            - Application Version: 1.0.0
            """
            
            st.code(system_info)
        except Exception as e:
            logger.error(f"Error rendering system information: {str(e)}")
            st.error(f"Failed to render system information: {str(e)}")
        
        # Clear data button (with confirmation)
        st.markdown("<h3>Data Management</h3>", unsafe_allow_html=True)
        
        if st.button("Clear All Application Data", key="clear_data"):
            # Confirmation
            if st.checkbox("I understand this will reset all targets, results, and settings", key="confirm_clear"):
                try:
                    # Reset all session state (except current page and theme)
                    current_page = st.session_state.current_page
                    current_theme = st.session_state.current_theme
                    
                    for key in list(st.session_state.keys()):
                        if key not in ['current_page', 'current_theme']:
                            del st.session_state[key]
                    
                    # Restore page and theme
                    st.session_state.current_page = current_page
                    st.session_state.current_theme = current_theme
                    
                    # Reinitialize session state
                    initialize_session_state()
                    
                    st.success("All application data has been cleared!")
                    logger.info("Application data cleared")
                    safe_rerun()
                except Exception as e:
                    logger.error(f"Error clearing application data: {str(e)}")
                    st.error(f"Failed to clear application data: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error rendering settings: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in settings: {str(e)}")

# ================= NEW PAGE RENDERERS =================

def render_file_import():
    """Render the file import page for multi-format support"""
    try:
        st.markdown("""
        <h2>Multi-Format Import</h2>
        <p>Import data in various formats for impact assessment</p>
        """, unsafe_allow_html=True)
        
        # File upload section
        st.markdown("<h3>Upload Files</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload File", 
            type=["json", "csv", "xlsx", "xls", "pdf", "xml", "yaml", "yml", "parquet", "h5", "arrow", "jsonl"],
            key="multi_format_upload"
        )
        
        if uploaded_file is not None:
            with st.spinner('Processing file...'):
                try:
                    # Process the file based on its type
                    processed_data = handle_multiple_file_formats(uploaded_file)
                    
                    if isinstance(processed_data, dict) and "error" in processed_data:
                        st.error(processed_data["error"])
                    else:
                        st.success(f"File '{uploaded_file.name}' processed successfully!")
                        
                        # Display different previews based on data type
                        if hasattr(processed_data, "head"):  # If it's a DataFrame
                            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
                            st.dataframe(processed_data.head(10))
                            
                            st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
                            st.write(processed_data.describe())
                            
                            # Store the data in session state
                            st.session_state.imported_data = processed_data
                            st.session_state.imported_file_name = uploaded_file.name
                            
                        elif isinstance(processed_data, dict):  # If it's a dictionary
                            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
                            st.json(processed_data)
                            
                            # Store the data in session state
                            st.session_state.imported_data = processed_data
                            st.session_state.imported_file_name = uploaded_file.name
                            
                        elif isinstance(processed_data, str):  # If it's text (like from PDF)
                            st.markdown("<h3>Content Preview</h3>", unsafe_allow_html=True)
                            st.text_area("Text Content", processed_data[:1000] + "..." if len(processed_data) > 1000 else processed_data, height=300)
                            
                            # Store the data in session state
                            st.session_state.imported_data = processed_data
                            st.session_state.imported_file_name = uploaded_file.name
                            
                        else:  # For other types
                            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
                            st.write(processed_data)
                            
                            # Store the data in session state
                            st.session_state.imported_data = processed_data
                            st.session_state.imported_file_name = uploaded_file.name
                        
                        # Show action buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Run Impact Assessment", key="run_impact_assessment", use_container_width=True):
                                st.session_state.current_page = "Environmental Impact"
                                safe_rerun()
                        
                        with col2:
                            if st.button("Run Bias Testing", key="run_bias_test", use_container_width=True):
                                st.session_state.current_page = "Bias Testing"
                                safe_rerun()
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logger.error(f"Error processing file: {str(e)}")
        
        # Information section
        st.markdown("<h3>Supported File Formats</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        This tool supports the following file formats:
        
        - **JSON** - JavaScript Object Notation
        - **CSV** - Comma-Separated Values
        - **Excel** - Microsoft Excel Spreadsheets (XLSX/XLS)
        - **PDF** - Portable Document Format
        - **XML** - eXtensible Markup Language
        - **YAML/YML** - YAML Ain't Markup Language
        - **Parquet** - Apache Parquet columnar storage
        - **HDF5** - Hierarchical Data Format version 5
        - **Arrow** - Apache Arrow columnar memory format
        - **JSONL** - JSON Lines format
        """)
        
        # Format usage examples
        with st.expander("Format Usage Examples"):
            st.markdown("""
            ### JSON
            Use JSON for configuration files, API responses, and structured data:
            ```json
            {
                "name": "Sample Model",
                "version": "1.0",
                "parameters": 7000000000,
                "metrics": {
                    "accuracy": 0.92,
                    "f1_score": 0.89
                }
            }
            ```
            
            ### CSV
            Use CSV for tabular data, such as model performance across different demographic groups:
            ```
            group,accuracy,precision,recall,f1_score
            male,0.94,0.92,0.91,0.915
            female,0.89,0.88,0.86,0.87
            non_binary,0.87,0.85,0.84,0.845
            ```
            
            ### YAML
            Use YAML for configuration files with nested structure:
            ```yaml
            model:
              name: GPT-4
              provider: OpenAI
              version: 1.0
            testing:
              vectors:
                - prompt_injection
                - jailbreaking
              intensity: high
            reporting:
              format: pdf
              details: full
            ```
            """)
        
    except Exception as e:
        logger.error(f"Error rendering file import: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in file import: {str(e)}")


def render_bias_testing():
    """Render the bias testing page with WhyLabs integration"""
    try:
        st.markdown("""
        <h2>AI Bias Testing</h2>
        <p>Analyze and mitigate bias in AI systems using WhyLabs</p>
        """, unsafe_allow_html=True)
        
        # Initialize WhyLabs bias tester if not already done
        if 'whylabs_bias_tester' not in st.session_state:
            st.session_state.whylabs_bias_tester = WhyLabsBiasTest()
        
        # Check if data is already imported
        if 'imported_data' in st.session_state and hasattr(st.session_state.imported_data, 'shape'):
            st.info(f"Using imported data: {st.session_state.imported_file_name} ({st.session_state.imported_data.shape[0]} rows, {st.session_state.imported_data.shape[1]} columns)")
            df = st.session_state.imported_data
            
            # Configuration section
            st.markdown("<h3>Bias Testing Configuration</h3>", unsafe_allow_html=True)
            
            # Select protected attributes (categorical columns)
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                protected_features = st.multiselect(
                    "Select Protected Attributes", 
                    categorical_columns,
                    key="protected_features"
                )
            
            with col2:
                # Select target column
                all_columns = df.columns.tolist()
                target_column = st.selectbox(
                    "Select Target Column (outcome)",
                    all_columns,
                    key="target_column"
                )
            
            # Dataset name
            dataset_name = st.text_input("Dataset Name", value=st.session_state.imported_file_name if 'imported_file_name' in st.session_state else "My Dataset", key="dataset_name")
            
            # Run analysis button
            if st.button("Run Bias Analysis", type="primary", use_container_width=True, key="run_bias_analysis"):
                if not protected_features:
                    st.error("Please select at least one protected attribute")
                elif not target_column:
                    st.error("Please select a target column")
                else:
                    with st.spinner('Running bias analysis...'):
                        # Run the bias analysis
                        bias_metrics = st.session_state.whylabs_bias_tester.analyze_bias(
                            df=df,
                            protected_features=protected_features,
                            target_column=target_column,
                            dataset_name=dataset_name
                        )
                        
                        if isinstance(bias_metrics, dict) and "error" in bias_metrics:
                            st.error(bias_metrics["error"])
                        else:
                            st.success("Bias analysis completed successfully!")
                            st.session_state.bias_results = bias_metrics
                            st.session_state.show_bias_results = True
                            safe_rerun()
            
            # Display results if available
            if st.session_state.get('show_bias_results', False):
                st.markdown("<h3>Bias Analysis Results</h3>", unsafe_allow_html=True)
                
                bias_results = st.session_state.bias_results
                
                for feature, metrics in bias_results.items():
                    with st.expander(f"Bias Analysis for {feature}", expanded=True):
                        # Show outcome rates by group
                        st.markdown(f"**Outcome Rates by {feature} Group:**")
                        
                        # Create a DataFrame for better display
                        import pandas as pd
                        outcomes_df = pd.DataFrame({
                            "Group": list(metrics["outcomes"].keys()),
                            "Outcome Rate": list(metrics["outcomes"].values())
                        })
                        
                        st.dataframe(outcomes_df)
                        
                        # Show disparities
                        st.markdown("**Disparities compared to highest outcome rate:**")
                        
                        disparities_df = pd.DataFrame({
                            "Group": list(metrics["disparities"].keys()),
                            "Disparity": list(metrics["disparities"].values())
                        })
                        
                        st.dataframe(disparities_df)
                        
                        # Visualize the disparities
                        try:
                            import plotly.express as px
                            
                            fig = px.bar(
                                disparities_df,
                                x="Group",
                                y="Disparity",
                                title=f"Outcome Disparities for {feature}",
                                color="Disparity",
                                color_continuous_scale="RdYlGn_r"  # Red for high disparity, green for low
                            )
                            
                            fig.update_layout(
                                yaxis_title="Disparity (lower is better)",
                                xaxis_title=feature,
                                margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=get_theme()["text"])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to create visualization: {str(e)}")
                
                # Provide recommendations based on findings
                st.markdown("<h3>Bias Mitigation Recommendations</h3>", unsafe_allow_html=True)
                
                # Calculate overall max disparity
                max_disparity_feature = max(bias_results.items(), key=lambda x: x[1]["max_disparity"])[0]
                max_disparity = bias_results[max_disparity_feature]["max_disparity"]
                
                # Generate recommendations based on disparity level
                if max_disparity > 0.2:
                    recommendation_level = "critical"
                elif max_disparity > 0.1:
                    recommendation_level = "significant"
                elif max_disparity > 0.05:
                    recommendation_level = "moderate"
                else:
                    recommendation_level = "minimal"
                
                recommendations = {
                    "critical": [
                        "Significant bias detected. Consider completely retraining your model with a carefully balanced dataset.",
                        "Implement pre-processing techniques to balance representation across protected groups.",
                        "Apply post-processing techniques to ensure equal error rates across groups.",
                        "Consider using adversarial debiasing during model training.",
                        "Conduct a comprehensive review of your data collection and labeling processes."
                    ],
                    "significant": [
                        "Implement fairness constraints during model training.",
                        "Review feature selection to identify and remove potential sources of bias.",
                        "Apply post-processing techniques to equalize error rates.",
                        "Augment training data for underrepresented groups.",
                        "Consider ensemble methods that can help mitigate bias."
                    ],
                    "moderate": [
                        "Monitor model performance across different demographic groups.",
                        "Implement fairness metrics in your model evaluation pipeline.",
                        "Consider data augmentation for minoritized groups.",
                        "Review feature importance and correlation with protected attributes."
                    ],
                    "minimal": [
                        "Continue monitoring model performance across groups.",
                        "Implement regular bias audits as part of model maintenance.",
                        "Document model limitations regarding potential biases."
                    ]
                }
                
                st.markdown(f"Based on our analysis, your model shows **{recommendation_level} bias issues**.")
                
                for recommendation in recommendations[recommendation_level]:
                    st.markdown(f"- {recommendation}")
                
                # Offer export options
                st.markdown("<h3>Export Results</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    try:
                        import json
                        
                        # Convert data to JSON
                        bias_report = {
                            "dataset": dataset_name,
                            "analysis_time": datetime.now().isoformat(),
                            "protected_features": protected_features,
                            "target_column": target_column,
                            "bias_metrics": bias_results,
                            "recommendation_level": recommendation_level,
                            "recommendations": recommendations[recommendation_level]
                        }
                        
                        st.download_button(
                            label="Download JSON Report",
                            data=json.dumps(bias_report, indent=2),
                            file_name=f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_bias_json"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare JSON export: {str(e)}")
                
                with col2:
                    try:
                        import pandas as pd
                        
                        # Create a summary DataFrame
                        summary_rows = []
                        
                        for feature, metrics in bias_results.items():
                            for group, disparity in metrics["disparities"].items():
                                summary_rows.append({
                                    "Protected_Feature": feature,
                                    "Group": group,
                                    "Outcome_Rate": metrics["outcomes"][group],
                                    "Disparity": disparity
                                })
                        
                        summary_df = pd.DataFrame(summary_rows)
                        csv = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV Summary",
                            data=csv,
                            file_name=f"bias_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_bias_csv"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare CSV export: {str(e)}")
                
                with col3:
                    try:
                        import io
                        import pandas as pd
                        
                        # Create Excel file in memory
                        output = io.BytesIO()
                        
                        # Create a Pandas Excel writer
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Create a summary sheet
                            summary_rows = []
                            
                            for feature, metrics in bias_results.items():
                                for group, disparity in metrics["disparities"].items():
                                    summary_rows.append({
                                        "Protected_Feature": feature,
                                        "Group": group,
                                        "Outcome_Rate": metrics["outcomes"][group],
                                        "Disparity": disparity
                                    })
                            
                            summary_df = pd.DataFrame(summary_rows)
                            summary_df.to_excel(writer, sheet_name='Bias Summary', index=False)
                            
                            # Add recommendations sheet
                            recs_data = {
                                'Recommendation': recommendations[recommendation_level]
                            }
                            recs_df = pd.DataFrame(recs_data)
                            recs_df.to_excel(writer, sheet_name='Recommendations', index=False)
                        
                        # Get the value from the BytesIO buffer
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_data,
                            file_name=f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_bias_excel"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare Excel export: {str(e)}")
        
        else:
            # Upload a CSV file directly on this page
            st.markdown("<h3>Upload Dataset</h3>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file", 
                type=["csv", "xlsx", "xls"],
                key="bias_testing_upload"
            )
            
            if uploaded_file is not None:
                with st.spinner('Loading dataset...'):
                    try:
                        # Determine file type and read accordingly
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        import pandas as pd
                        if file_extension == 'csv':
                            df = pd.read_csv(uploaded_file)
                        elif file_extension in ['xlsx', 'xls']:
                            df = pd.read_excel(uploaded_file)
                        else:
                            st.error("Unsupported file format. Please upload a CSV or Excel file.")
                            return
                        
                        # Store the data
                        st.session_state.imported_data = df
                        st.session_state.imported_file_name = uploaded_file.name
                        
                        st.success(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                        safe_rerun()
                    
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
            
            # Option to use sample data
            st.markdown("<h3>Or Use Sample Dataset</h3>", unsafe_allow_html=True)
            
            if st.button("Load Sample Dataset", key="load_sample_dataset"):
                with st.spinner('Loading sample dataset...'):
                    try:
                        # Create a sample dataset with potential bias
                        import pandas as pd
                        import numpy as np
                        
                        # Set seed for reproducibility
                        np.random.seed(42)
                        
                        # Create sample data
                        n_samples = 1000
                        
                        # Generate demographic features
                        gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
                        age_group = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], size=n_samples)
                        ethnicity = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], size=n_samples, p=[0.6, 0.2, 0.15, 0.05])
                        
                        # Generate features
                        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_samples)
                        experience = np.random.randint(0, 20, size=n_samples)
                        
                        # Create biased outcomes
                        # Male applicants have a higher chance of approval
                        gender_bias = (gender == 'Male') * 0.2
                        
                        # Group A has higher approval rate
                        ethnicity_bias = np.zeros(n_samples)
                        ethnicity_bias[ethnicity == 'Group A'] = 0.1
                        ethnicity_bias[ethnicity == 'Group D'] = -0.15
                        
                        # Base approval probability
                        base_prob = 0.5
                        approval_prob = base_prob + gender_bias + ethnicity_bias
                        
                        # Ensure probabilities are between 0 and 1
                        approval_prob = np.clip(approval_prob, 0, 1)
                        
                        # Generate approval decisions
                        approved = np.random.binomial(1, approval_prob)
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'Gender': gender,
                            'Age_Group': age_group,
                            'Ethnicity': ethnicity,
                            'Education': education,
                            'Experience_Years': experience,
                            'Approved': approved
                        })
                        
                        # Store the data
                        st.session_state.imported_data = df
                        st.session_state.imported_file_name = "sample_biased_dataset.csv"
                        
                        st.success("Sample dataset loaded successfully")
                        safe_rerun()
                    
                    except Exception as e:
                        st.error(f"Error creating sample dataset: {str(e)}")
            
            # Provide some information
            st.markdown("""
            ### About Bias Testing
            
            Bias testing helps identify and mitigate unfair discrimination in AI systems. Common biases include:
            
            - **Demographic bias**: When the model performs differently across demographic groups
            - **Representation bias**: When training data doesn't adequately represent all groups
            - **Measurement bias**: When the choice of what to measure creates disparities
            - **Aggregation bias**: When assumptions that apply to the overall population don't apply to subgroups
            
            The WhyLabs integration helps you identify these biases and provides mitigation strategies.
            """)
    
    except Exception as e:
        logger.error(f"Error rendering bias testing: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in bias testing: {str(e)}")


def render_environmental_impact():
    """Render the environmental impact assessment page"""
    try:
        st.markdown("""
        <h2>Environmental Impact Assessment</h2>
        <p>Analyze and mitigate the carbon footprint of your AI systems</p>
        """, unsafe_allow_html=True)
        
        # Initialize carbon tracker if not already done
        if 'carbon_tracker' not in st.session_state:
            st.session_state.carbon_tracker = CarbonImpactTracker()
            st.session_state.carbon_tracker_initialized = False
        
        # Create tabs for different functionality
        tabs = st.tabs(["Carbon Measurement", "Model Analysis", "Optimization Strategies"])
        
        with tabs[0]:
            st.markdown("<h3>Carbon Emission Tracking</h3>", unsafe_allow_html=True)
            
            # Initialize tracker if needed
            if not st.session_state.carbon_tracker_initialized:
                project_name = st.text_input("Project Name", value="AI Security Assessment", key="carbon_project_name")
                
                if st.button("Initialize Carbon Tracker", key="init_carbon_tracker"):
                    with st.spinner("Initializing tracker..."):
                        success = st.session_state.carbon_tracker.initialize_tracker(project_name)
                        
                        if success:
                            st.session_state.carbon_tracker_initialized = True
                            st.success("Carbon tracker initialized successfully!")
                            safe_rerun()
                        else:
                            st.error("Failed to initialize carbon tracker. Please check logs for details.")
            else:
                # Tracking controls
                if not st.session_state.get("carbon_tracking_active", False):
                    if st.button("Start Carbon Tracking", key="start_carbon_tracking", type="primary"):
                        success = st.session_state.carbon_tracker.start_tracking()
                        
                        if success:
                            st.session_state.carbon_tracking_active = True
                            st.success("Carbon tracking started!")
                            safe_rerun()
                        else:
                            st.error("Failed to start carbon tracking. Please check logs for details.")
                else:
                    if st.button("Stop Carbon Tracking", key="stop_carbon_tracking", type="primary"):
                        emissions = st.session_state.carbon_tracker.stop_tracking()
                        
                        st.session_state.carbon_tracking_active = False
                        st.success(f"Carbon tracking stopped! Measured: {emissions:.6f} kg CO2eq")
                        
                        # Store the last measurement
                        if 'carbon_measurements' not in st.session_state:
                            st.session_state.carbon_measurements = []
                        
                        st.session_state.carbon_measurements.append({
                            "timestamp": datetime.now().isoformat(),
                            "emissions_kg": emissions
                        })
                        
                        safe_rerun()
                
                # Display tracking status
                if st.session_state.get("carbon_tracking_active", False):
                    st.info("Carbon tracking is active. Run your AI operations and stop tracking when finished.")
                
                # Display measurements
                st.markdown("<h4>Emission Measurements</h4>", unsafe_allow_html=True)
                
                total_emissions = st.session_state.carbon_tracker.get_total_emissions()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Emissions", f"{total_emissions:.6f} kg CO2eq")
                
                with col2:
                    # Convert to equivalent metrics
                    miles_driven = total_emissions * 2.4  # ~2.4 miles per kg CO2
                    st.metric("Equivalent Car Miles", f"{miles_driven:.2f} miles")
                
                with col3:
                    # Trees needed to offset
                    trees_needed = total_emissions * 0.06  # ~0.06 trees per kg CO2 per year
                    st.metric("Trees Needed (1 year)", f"{trees_needed:.2f} trees")
                
                # Show all measurements
                if 'carbon_measurements' in st.session_state and st.session_state.carbon_measurements:
                    st.markdown("<h4>Measurement History</h4>", unsafe_allow_html=True)
                    
                    import pandas as pd
                    
                    measurements_df = pd.DataFrame(st.session_state.carbon_measurements)
                    st.dataframe(measurements_df)
                    
                    # Create a chart
                    try:
                        import plotly.express as px
                        
                        measurements_df['timestamp'] = pd.to_datetime(measurements_df['timestamp'])
                        
                        fig = px.line(
                            measurements_df,
                            x='timestamp',
                            y='emissions_kg',
                            title='Carbon Emissions Over Time',
                            labels={'timestamp': 'Time', 'emissions_kg': 'Emissions (kg CO2eq)'}
                        )
                        
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=get_theme()["text"])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to create chart: {str(e)}")
        
        with tabs[1]:
            st.markdown("<h3>AI Model Carbon Footprint Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name", value="", key="model_name")
                model_parameters = st.number_input("Model Parameters (millions)", min_value=0.1, max_value=1000000.0, value=1.0, key="model_parameters")
                training_hours = st.number_input("Training Hours", min_value=0, max_value=10000, value=24, key="training_hours")
            
            with col2:
                hardware_options = ["CPU", "GPU - Consumer", "GPU - Data Center", "TPU", "Custom ASIC"]
                hardware_type = st.selectbox("Hardware Type", hardware_options, key="hardware_type")
                daily_inferences = st.number_input("Daily Inferences", min_value=0, max_value=1000000000, value=10000, key="daily_inferences")
                hosting_region = st.selectbox("Hosting Region", [
                    "North America", "Europe", "Asia Pacific", "South America", "Africa", "Middle East"
                ], key="hosting_region")
            
            if st.button("Calculate Carbon Footprint", key="calc_footprint", type="primary"):
                with st.spinner("Calculating carbon footprint..."):
                    try:
                        # Carbon intensity by region (kg CO2 per kWh)
                        region_carbon_intensity = {
                            "North America": 0.38,
                            "Europe": 0.28,
                            "Asia Pacific": 0.55,
                            "South America": 0.21,
                            "Africa": 0.47,
                            "Middle East": 0.60
                        }
                        
                        # Power consumption by hardware type (W)
                        hardware_power = {
                            "CPU": 150,
                            "GPU - Consumer": 250,
                            "GPU - Data Center": 300,
                            "TPU": 200,
                            "Custom ASIC": 60
                        }
                        
                        # Energy efficiency by hardware (inferences per joule)
                        hardware_efficiency = {
                            "CPU": 5,
                            "GPU - Consumer": 20,
                            "GPU - Data Center": 50,
                            "TPU": 80,
                            "Custom ASIC": 120
                        }
                        
                        # Calculate training emissions
                        hardware_power_kw = hardware_power[hardware_type] / 1000  # Convert W to kW
                        training_energy_kwh = hardware_power_kw * training_hours
                        training_emissions = training_energy_kwh * region_carbon_intensity[hosting_region]
                        
                        # Calculate inference emissions (daily)
                        inference_power_per_request = 1 / hardware_efficiency[hardware_type]  # Joules per inference
                        daily_inference_energy_kwh = (inference_power_per_request * daily_inferences) / (3.6 * 10**6)  # Convert J to kWh
                        daily_inference_emissions = daily_inference_energy_kwh * region_carbon_intensity[hosting_region]
                        
                        # Yearly inference emissions
                        yearly_inference_emissions = daily_inference_emissions * 365
                        
                        # Total first-year emissions
                        total_first_year = training_emissions + yearly_inference_emissions
                        
                        # Store results
                        st.session_state.model_analysis = {
                            "model_name": model_name,
                            "model_parameters": model_parameters,
                            "training_emissions": training_emissions,
                            "daily_inference_emissions": daily_inference_emissions,
                            "yearly_inference_emissions": yearly_inference_emissions,
                            "total_first_year": total_first_year,
                            "hardware_type": hardware_type,
                            "hosting_region": hosting_region
                        }
                        
                        # Show success message
                        st.success("Carbon footprint calculation complete!")
                        safe_rerun()
                    
                    except Exception as e:
                        st.error(f"Failed to calculate carbon footprint: {str(e)}")
            
            # Display results if available
            if 'model_analysis' in st.session_state:
                results = st.session_state.model_analysis
                
                st.markdown(f"<h4>Carbon Footprint: {results['model_name']}</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Emissions", f"{results['training_emissions']:.2f} kg CO2eq")
                
                with col2:
                    st.metric("Daily Inference", f"{results['daily_inference_emissions']:.4f} kg CO2eq")
                
                with col3:
                    st.metric("Yearly Emissions", f"{results['total_first_year']:.2f} kg CO2eq")
                
                # Chart comparing training vs inference
                try:
                    import plotly.express as px
                    import pandas as pd
                    
                    # Prepare data
                    emission_types = ["Training", "Inference (1 year)"]
                    emission_values = [results['training_emissions'], results['yearly_inference_emissions']]
                    
                    emissions_df = pd.DataFrame({
                        "Emission Source": emission_types,
                        "CO2 Emissions (kg)": emission_values
                    })
                    
                    fig = px.bar(
                        emissions_df,
                        y="Emission Source",
                        x="CO2 Emissions (kg)",
                        title="Carbon Emissions Breakdown",
                        orientation='h',
                        color="Emission Source",
                        color_discrete_map={"Training": get_theme()["primary"], "Inference (1 year)": get_theme()["secondary"]}
                    )
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=get_theme()["text"])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to create chart: {str(e)}")
                
                # Mitigation strategies
                st.markdown("<h4>Mitigation Strategies</h4>", unsafe_allow_html=True)
                
                # Calculate potential savings
                current_region = results['hosting_region']
                current_intensity = {
                    "North America": 0.38,
                    "Europe": 0.28,
                    "Asia Pacific": 0.55,
                    "South America": 0.21,
                    "Africa": 0.47,
                    "Middle East": 0.60
                }[current_region]
                
                min_region = min({"North America": 0.38, "Europe": 0.28, "Asia Pacific": 0.55, "South America": 0.21, "Africa": 0.47, "Middle East": 0.60}.items(), key=lambda x: x[1])
                
                region_savings = (current_intensity - min_region[1]) / current_intensity * 100
                
                # Distillation savings (typically 70-90% reduction)
                distillation_savings = 80  # percentage
                
                # Efficient hardware (depends on current choice)
                current_hardware = results['hardware_type']
                best_efficiency = max({"CPU": 5, "GPU - Consumer": 20, "GPU - Data Center": 50, "TPU": 80, "Custom ASIC": 120}.items(), key=lambda x: x[1])
                
                current_efficiency = {"CPU": 5, "GPU - Consumer": 20, "GPU - Data Center": 50, "TPU": 80, "Custom ASIC": 120}[current_hardware]
                hardware_savings = (1 - current_efficiency / best_efficiency[1]) * 100
                
                # Display strategies with expected savings
                strategies = [
                    {
                        "name": "Model Distillation",
                        "description": "Create a smaller, efficient version of your model",
                        "savings": f"{distillation_savings}% reduction",
                        "effort": "High"
                    },
                    {
                        "name": "Low-Carbon Region",
                        "description": f"Move from {current_region} to {min_region[0]}",
                        "savings": f"{region_savings:.1f}% reduction",
                        "effort": "Medium"
                    },
                    {
                        "name": "Efficient Hardware",
                        "description": f"Move from {current_hardware} to {best_efficiency[0]}",
                        "savings": f"{hardware_savings:.1f}% reduction" if hardware_savings > 0 else "Already optimal",
                        "effort": "Medium" if hardware_savings > 0 else "N/A"
                    },
                    {
                        "name": "Quantization",
                        "description": "Reduce model precision (e.g., FP32 to INT8)",
                        "savings": "60-75% reduction",
                        "effort": "Medium"
                    },
                    {
                        "name": "Caching",
                        "description": "Cache frequent inference results",
                        "savings": "Varies by usage pattern",
                        "effort": "Low"
                    }
                ]
                
                for i, strategy in enumerate(strategies):
                    with st.expander(f"{i+1}. {strategy['name']} - {strategy['savings']}", expanded=i==0):
                        st.markdown(f"**Description:** {strategy['description']}")
                        st.markdown(f"**Implementation Effort:** {strategy['effort']}")
                        
                        if strategy['name'] == "Model Distillation":
                            st.markdown("""
                            **Implementation Steps:**
                            1. Train a student model using the original model's outputs
                            2. Optimize the student model architecture
                            3. Fine-tune the student model
                            4. Validate performance meets requirements
                            """)
                        elif strategy['name'] == "Low-Carbon Region":
                            st.markdown("""
                            **Implementation Steps:**
                            1. Identify data centers in low-carbon regions
                            2. Evaluate compliance and data sovereignty requirements
                            3. Plan migration strategy
                            4. Deploy and validate
                            """)
                        elif strategy['name'] == "Efficient Hardware":
                            st.markdown("""
                            **Implementation Steps:**
                            1. Benchmark model on different hardware
                            2. Optimize model for target hardware
                            3. Evaluate cost-performance tradeoffs
                            4. Deploy and monitor
                            """)
        
        with tabs[2]:
            st.markdown("<h3>Carbon Optimization Strategies</h3>", unsafe_allow_html=True)
            
            # Describe different strategies
            strategies = [
                {
                    "name": "Model Architecture Optimization",
                    "description": "Design efficient model architectures that require less computation while maintaining accuracy.",
                    "techniques": [
                        "Pruning - Remove unnecessary connections or neurons",
                        "Knowledge Distillation - Train smaller models using larger models as teachers",
                        "Neural Architecture Search - Automatically find efficient architectures",
                        "Sparsity - Encourage sparse activations and weights"
                    ],
                    "case_studies": [
                        "DistilBERT - 40% smaller, 60% faster, 97% of BERT's performance",
                        "MobileNet - Efficient CNN for mobile and edge devices",
                        "EfficientNet - Scaled model achieving state-of-the-art performance with fewer parameters"
                    ],
                    "tools": [
                        "TensorFlow Model Optimization Toolkit",
                        "PyTorch Pruning API",
                        "Microsoft's Neural Network Intelligence (NNI)",
                        "Google AutoML"
                    ]
                },
                {
                    "name": "Quantization and Precision Reduction",
                    "description": "Reduce the numerical precision of model weights and operations.",
                    "techniques": [
                        "Post-training quantization - Convert trained model to lower precision",
                        "Quantization-aware training - Train with simulated quantization",
                        "Mixed-precision training - Use different precisions for different operations",
                        "Binary/ternary networks - Use 1-bit or 2-bit weights"
                    ],
                    "case_studies": [
                        "BERT-INT8 - 4x faster inference with minimal accuracy loss",
                        "NVIDIA TensorRT quantization - Up to 3x performance improvement",
                        "Binarized Neural Networks - 7x energy efficiency improvement"
                    ],
                    "tools": [
                        "TensorFlow Lite Converter",
                        "PyTorch Quantization",
                        "NVIDIA TensorRT",
                        "Intel Neural Compressor"
                    ]
                },
                {
                    "name": "Efficient Training Strategies",
                    "description": "Optimize the training process to reduce computational requirements.",
                    "techniques": [
                        "Transfer learning - Start from pre-trained models",
                        "Early stopping - Terminate training when validation performance plateaus",
                        "Learning rate scheduling - Optimize convergence speed",
                        "Gradient accumulation - Reduce memory requirements"
                    ],
                    "case_studies": [
                        "GPT-3 Fine-tuning - 100x less computation than training from scratch",
                        "ELECTRA - 4x more efficient training than BERT",
                        "Progressive resizing - Train on smaller images first, then larger ones"
                    ],
                    "tools": [
                        "Hugging Face Accelerate",
                        "DeepSpeed",
                        "PyTorch Lightning",
                        "Weights & Biases for experiment tracking"
                    ]
                },
                {
                    "name": "Carbon-Aware Computing",
                    "description": "Schedule and locate computation based on carbon intensity of electricity.",
                    "techniques": [
                        "Carbon-aware scheduling - Run compute jobs when carbon intensity is low",
                        "Geographic optimization - Choose regions with low-carbon electricity",
                        "Demand shifting - Move non-urgent workloads to optimal times",
                        "Renewable energy matching - Time workloads with renewable generation"
                    ],
                    "case_studies": [
                        "Google Carbon-Intelligent Computing - Shifted 35% of non-urgent compute",
                        "Microsoft Azure's carbon-aware migration - 60% reduction in carbon emissions",
                        "Flexible GPT training - 30% emission reduction through scheduling"
                    ],
                    "tools": [
                        "WattTime API",
                        "Electricity Maps API",
                        "Google Carbon-Intelligent Platform",
                        "CodeCarbon"
                    ]
                }
            ]
            
            # Create expandable sections for each strategy
            for i, strategy in enumerate(strategies):
                with st.expander(f"{i+1}. {strategy['name']}", expanded=i==0):
                    st.markdown(f"**{strategy['description']}**")
                    
                    st.markdown("#### Key Techniques")
                    for technique in strategy['techniques']:
                        st.markdown(f"- {technique}")
                    
                    st.markdown("#### Case Studies")
                    for case in strategy['case_studies']:
                        st.markdown(f"- {case}")
                    
                    st.markdown("#### Tools & Resources")
                    for tool in strategy['tools']:
                        st.markdown(f"- {tool}")
            
            # Carbon reduction estimator
            st.markdown("<h3>Carbon Reduction Estimator</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                baseline_emissions = st.number_input("Baseline Annual Emissions (kg CO2eq)", min_value=0.0, value=1000.0, key="baseline_emissions")
                apply_distillation = st.checkbox("Apply Model Distillation (70-90% reduction)", value=True, key="apply_distillation")
                apply_quantization = st.checkbox("Apply Quantization (60-75% reduction)", value=True, key="apply_quantization")
            
            with col2:
                carbon_aware_scheduling = st.checkbox("Carbon-Aware Scheduling (15-40% reduction)", value=True, key="carbon_aware")
                efficient_hardware = st.checkbox("Use Efficient Hardware (20-60% reduction)", value=True, key="efficient_hardware")
                apply_caching = st.checkbox("Implement Result Caching (10-30% reduction)", value=False, key="apply_caching")
            
            if st.button("Calculate Potential Savings", key="calc_savings", type="primary"):
                try:
                    # Calculate combined reductions
                    reduction_factors = []
                    
                    if apply_distillation:
                        reduction_factors.append(0.8)  # 80% reduction
                    
                    if apply_quantization:
                        # Apply to remaining emissions after previous reductions
                        remaining = 1.0
                        for factor in reduction_factors:
                            remaining *= (1 - factor)
                        
                        reduction_factors.append(0.7 * remaining)  # 70% reduction of remaining
                    
                    if carbon_aware_scheduling:
                        # Apply to remaining emissions
                        remaining = 1.0
                        for factor in reduction_factors:
                            remaining *= (1 - factor)
                        
                        reduction_factors.append(0.25 * remaining)  # 25% reduction of remaining
                    
                    if efficient_hardware:
                        # Apply to remaining emissions
                        remaining = 1.0
                        for factor in reduction_factors:
                            remaining *= (1 - factor)
                        
                        reduction_factors.append(0.4 * remaining)  # 40% reduction of remaining
                    
                    if apply_caching:
                        # Apply to remaining emissions
                        remaining = 1.0
                        for factor in reduction_factors:
                            remaining *= (1 - factor)
                        
                        reduction_factors.append(0.2 * remaining)  # 20% reduction of remaining
                    
                    # Calculate total reduction
                    total_reduction = sum(reduction_factors)
                    
                    # Calculate new emissions
                    new_emissions = baseline_emissions * (1 - total_reduction)
                    emissions_saved = baseline_emissions - new_emissions
                    
                    # Store results
                    st.session_state.emission_reduction = {
                        "baseline": baseline_emissions,
                        "new_emissions": new_emissions,
                        "emissions_saved": emissions_saved,
                        "percentage_reduction": total_reduction * 100
                    }
                    
                    # Show success message
                    st.success(f"Potential carbon reduction: {total_reduction*100:.1f}%")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to calculate savings: {str(e)}")
            
            # Display results if available
            if 'emission_reduction' in st.session_state:
                results = st.session_state.emission_reduction
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Baseline Emissions", f"{results['baseline']:.2f} kg CO2eq")
                
                with col2:
                    st.metric("New Emissions", f"{results['new_emissions']:.2f} kg CO2eq")
                
                with col3:
                    st.metric("Reduction", f"{results['percentage_reduction']:.1f}%")
                
                # Visualize comparison
                try:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=["Baseline", "Optimized"],
                        y=[results['baseline'], results['new_emissions']],
                        marker_color=[get_theme()["error"], get_theme()["primary"]]
                    ))
                    
                    fig.update_layout(
                        title="Emission Comparison",
                        yaxis_title="CO2 Emissions (kg)",
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=get_theme()["text"])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to create chart: {str(e)}")
                
                # Environmental equivalents
                st.markdown("<h4>Environmental Impact</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Car miles equivalent (1 kg CO2 ~ 2.4 miles)
                    car_miles = results['emissions_saved'] * 2.4
                    st.metric("Car Miles Saved", f"{car_miles:.0f} miles")
                
                with col2:
                    # Trees absorbing carbon (1 tree absorbs ~22 kg CO2 per year)
                    trees = results['emissions_saved'] / 22
                    st.metric("Equivalent Trees", f"{trees:.1f} trees/year")
                
                with col3:
                    # Smartphone charges (1 kg CO2 ~ 126 charges)
                    phone_charges = results['emissions_saved'] * 126
                    st.metric("Phone Charges", f"{phone_charges:.0f} charges")
    
    except Exception as e:
        logger.error(f"Error rendering environmental impact: {str(e)}")
        logger.debug(traceback.format_exc())
        st.error(f"Error in environmental impact assessment: {str(e)}")

# Main application
def main():
    """Main application entry point with error handling"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Clean up threads
        cleanup_threads()
        
        # Apply CSS
        st.markdown(load_css(), unsafe_allow_html=True)
        
        # Show error message if exists
        if st.session_state.error_message:
            st.markdown(f"""
            <div class="error-message">
                <strong>Error:</strong> {st.session_state.error_message}
            </div>
            """, unsafe_allow_html=True)
            
            # Add button to clear error
            if st.button("Clear Error"):
                st.session_state.error_message = None
                safe_rerun()
        
        # Render sidebar
        sidebar_navigation()
        
        # Render header
        render_header()
        
        # Render content based on current page
        if st.session_state.current_page == "Dashboard":
            render_dashboard()
        elif st.session_state.current_page == "Target Management":
            render_target_management()
        elif st.session_state.current_page == "Test Configuration":
            render_test_configuration()
        elif st.session_state.current_page == "Run Assessment":
            render_run_assessment()
        elif st.session_state.current_page == "Results Analyzer":
            render_results_analyzer()
        elif st.session_state.current_page == "Ethical AI Testing":
            render_ethical_ai_testing()
        elif st.session_state.current_page == "Environmental Impact":   # New page
            render_environmental_impact()
        elif st.session_state.current_page == "Bias Testing":           # New page
            render_bias_testing()
        elif st.session_state.current_page == "Multi-Format Import":    # New page
            render_file_import()
        elif st.session_state.current_page == "High-Volume Testing":
            render_high_volume_testing()
        elif st.session_state.current_page == "Settings":
            render_settings()
        else:
            # Default to dashboard if invalid page
            logger.warning(f"Invalid page requested: {st.session_state.current_page}")
            st.session_state.current_page = "Dashboard"
            render_dashboard()
    
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        logger.critical(traceback.format_exc())
        st.error(f"Critical application error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
