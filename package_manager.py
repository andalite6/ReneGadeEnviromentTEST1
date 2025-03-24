# Add this code to your application to enable dynamic loading of PyPI packages

import importlib
import subprocess
import sys
import logging
import importlib.util
from typing import Optional, Union, List, Tuple, Dict, Any

logger = logging.getLogger("PackageManager")

class PackageManager:
    """Utility class for dynamically installing and importing packages at runtime"""
    
    @staticmethod
    def is_package_installed(package_name: str) -> bool:
        """Check if a package is already installed"""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False
    
    @staticmethod
    def install_package(package_name: str, version: Optional[str] = None, upgrade: bool = False) -> bool:
        """Install a package using pip"""
        try:
            package_spec = package_name
            if version:
                package_spec = f"{package_name}=={version}"
            
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package_spec)
            
            logger.info(f"Installing package: {package_spec}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install {package_spec}: {result.stderr}")
                return False
            
            logger.info(f"Successfully installed {package_spec}")
            return True
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {str(e)}")
            return False
    
    @staticmethod
    def import_package(package_name: str, as_name: Optional[str] = None) -> Any:
        """Import a package, install it if necessary"""
        try:
            # Check if package is already imported
            if package_name in sys.modules:
                module = sys.modules[package_name]
            else:
                # Try to import the package
                try:
                    module = importlib.import_module(package_name)
                except ImportError:
                    # If import fails, try to install the package and then import
                    success = PackageManager.install_package(package_name)
                    if not success:
                        raise ImportError(f"Could not install package {package_name}")
                    
                    # Now try importing again
                    module = importlib.import_module(package_name)
            
            # If as_name is provided, add it to sys.modules under that name
            if as_name is not None and as_name != package_name:
                sys.modules[as_name] = module
            
            return module
        except Exception as e:
            logger.error(f"Error importing package {package_name}: {str(e)}")
            raise

# Example usage:
"""
# Import a single package
try:
    pandas = PackageManager.import_package('pandas')
    
    # Use pandas as normal
    df = pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(df)
except Exception as e:
    st.error(f"Failed to import pandas: {str(e)}")

# Import a package with a specific version
try:
    numpy = PackageManager.import_package('numpy', as_name='np')
    
    # Now use numpy as np
    arr = np.array([1, 2, 3])
    print(arr)
except Exception as e:
    st.error(f"Failed to import numpy: {str(e)}")

# Try importing a package that may not be installed
try:
    # This will automatically install scikit-learn if not already installed
    sklearn = PackageManager.import_package('scikit-learn')
    
    # Use sklearn
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
except Exception as e:
    st.error(f"Failed to import scikit-learn: {str(e)}")
"""
# Add this to the settings page to allow users to install packages directly from the UI

def render_package_manager():
    """Render the package manager section in the settings page"""
    st.markdown("<h3>Package Manager</h3>", unsafe_allow_html=True)
    
    # Package installation form
    with st.form("install_package_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            package_name = st.text_input("Package Name", key="package_name_input", 
                                       help="Enter the name of a PyPI package to install")
        
        with col2:
            package_version = st.text_input("Version (optional)", key="package_version_input",
                                          help="Leave blank for latest version")
        
        upgrade_package = st.checkbox("Upgrade if already installed", value=False, key="upgrade_package")
        
        submit_button = st.form_submit_button("Install Package")
        
        if submit_button:
            if not package_name:
                st.error("Please enter a package name")
            else:
                with st.spinner(f"Installing {package_name}..."):
                    try:
                        success = PackageManager.install_package(
                            package_name, 
                            version=package_version if package_version else None,
                            upgrade=upgrade_package
                        )
                        
                        if success:
                            st.success(f"Successfully installed {package_name}" + 
                                     (f" {package_version}" if package_version else ""))
                            
                            # Store the installed package in session state
                            if 'installed_packages' not in st.session_state:
                                st.session_state.installed_packages = []
                            
                            # Add to list if not already there
                            package_entry = f"{package_name}" + (f"=={package_version}" if package_version else "")
                            if package_entry not in st.session_state.installed_packages:
                                st.session_state.installed_packages.append(package_entry)
                        else:
                            st.error(f"Failed to install {package_name}. Check the logs for details.")
                    except Exception as e:
                        st.error(f"Error installing package: {str(e)}")
    
    # Display installed packages
    if 'installed_packages' in st.session_state and st.session_state.installed_packages:
        st.markdown("<h4>Installed Packages</h4>", unsafe_allow_html=True)
        
        for i, package in enumerate(st.session_state.installed_packages):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.text(package)
            
            with col2:
                if st.button("Remove", key=f"remove_package_{i}"):
                    try:
                        # This doesn't actually uninstall the package, just removes it from the list
                        st.session_state.installed_packages.remove(package)
                        st.success(f"Removed {package} from list")
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Error removing package: {str(e)}")
    
    # Package import tester
    st.markdown("<h4>Test Package Import</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_package = st.text_input("Package to import", key="test_package_input")
    
    with col2:
        alias = st.text_input("Import as (optional)", key="import_alias")
    
    if st.button("Test Import", key="test_import"):
        if not test_package:
            st.error("Please enter a package name to test")
        else:
            with st.spinner(f"Importing {test_package}..."):
                try:
                    module = PackageManager.import_package(
                        test_package,
                        as_name=alias if alias else None
                    )
                    
                    # Get package version
                    version = getattr(module, "__version__", "unknown version")
                    
                    st.success(f"Successfully imported {test_package} ({version})")
                    
                    # Display some helpful information about the package
                    if hasattr(module, "__doc__") and module.__doc__:
                        with st.expander("Package Documentation"):
                            st.code(module.__doc__)
                except Exception as e:
                    st.error(f"Failed to import {test_package}: {str(e)}")

# Add this to the `render_settings` function:
"""
    # Add the package manager section
    st.markdown("---")
    render_package_manager()
"""

# Also add the PackageManager class import at the top of your file
"""
from package_manager import PackageManager  # Import your PackageManager class
"""
