import importlib
import subprocess
import sys
import logging
import importlib.util
from typing import Optional, Union, List, Tuple, Dict, Any
import re

logger = logging.getLogger("PackageManager")

class PackageManager:
    """Utility class for dynamically installing and importing packages at runtime with enhanced error handling"""
    
    @staticmethod
    def is_package_installed(package_name: str) -> bool:
        """Check if a package is already installed"""
        try:
            # Handle package names with extras or version specifiers
            base_package = package_name.split('[')[0].split('==')[0].split('>=')[0].split('<=')[0].strip()
            spec = importlib.util.find_spec(base_package)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False
    
    @staticmethod
    def install_package(package_name: str, version: Optional[str] = None, upgrade: bool = False) -> bool:
        """Install a package using pip with enhanced error handling"""
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
            
            # Check for specific error patterns
            if result.returncode != 0:
                error_msg = result.stderr
                
                # Check for no matching distribution
                if "No matching distribution found for" in error_msg:
                    package_in_error = re.search(r"No matching distribution found for ([^\s]+)", error_msg)
                    if package_in_error:
                        problem_package = package_in_error.group(1)
                        logger.error(f"Package '{problem_package}' not found on PyPI or incompatible with Python version")
                        return False
                
                # Check for Python version compatibility
                if "Requires-Python" in error_msg:
                    logger.error(f"Package '{package_spec}' is not compatible with the current Python version")
                    return False
                
                # Generic error
                logger.error(f"Failed to install {package_spec}: {error_msg}")
                return False
            
            logger.info(f"Successfully installed {package_spec}")
            return True
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {str(e)}")
            return False
    
    @staticmethod
    def find_compatible_version(package_name: str, max_attempts: int = 5) -> Optional[str]:
        """Try to find a compatible version of a package"""
        try:
            # Get available versions
            cmd = [sys.executable, "-m", "pip", "index", "versions", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
                
            # Parse output to find available versions
            available_versions = []
            for line in result.stdout.splitlines():
                if "Available versions:" in line:
                    versions_str = line.split("Available versions:")[1].strip()
                    available_versions = [v.strip() for v in versions_str.split(",")]
                    break
            
            # Try the most recent versions first (usually listed in descending order)
            attempts = 0
            for version in available_versions[:max_attempts]:
                # Try to install this specific version
                cmd = [sys.executable, "-m", "pip", "install", f"{package_name}=={version}"]
                install_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if install_result.returncode == 0:
                    return version
                
                attempts += 1
                if attempts >= max_attempts:
                    break
            
            return None
        except Exception as e:
            logger.error(f"Error finding compatible version for {package_name}: {str(e)}")
            return None
    
    @staticmethod
    def import_package(package_name: str, as_name: Optional[str] = None, 
                      auto_install: bool = True, attempt_alternate: bool = True) -> Any:
        """
        Import a package, install it if necessary, with enhanced error handling
        
        Args:
            package_name: The name of the package to import
            as_name: Optional alias to import the package as
            auto_install: Whether to attempt installation if package is not found
            attempt_alternate: Whether to try finding alternative versions if installation fails
            
        Returns:
            The imported module
        """
        try:
            # Extract the base package name (without version/extras)
            base_package = package_name.split('[')[0].split('==')[0].split('>=')[0].split('<=')[0].strip()
            
            # Check if package is already imported
            if base_package in sys.modules:
                module = sys.modules[base_package]
            else:
                # Try to import the package
                try:
                    module = importlib.import_module(base_package)
                except ImportError:
                    if not auto_install:
                        raise ImportError(f"Package {base_package} not found and auto_install is disabled")
                    
                    # If import fails, try to install the package and then import
                    success = PackageManager.install_package(package_name)
                    
                    if not success and attempt_alternate:
                        # Try to find a compatible version
                        logger.info(f"Attempting to find compatible version for {base_package}")
                        compatible_version = PackageManager.find_compatible_version(base_package)
                        
                        if compatible_version:
                            logger.info(f"Found compatible version {compatible_version} for {base_package}")
                            success = PackageManager.install_package(f"{base_package}=={compatible_version}")
                        else:
                            # Try installing just the base package without version constraints
                            success = PackageManager.install_package(base_package)
                    
                    if not success:
                        raise ImportError(f"Could not install package {package_name}")
                    
                    # Now try importing again
                    module = importlib.import_module(base_package)
            
            # If as_name is provided, add it to sys.modules under that name
            if as_name is not None and as_name != base_package:
                sys.modules[as_name] = module
            
            return module
        except Exception as e:
            logger.error(f"Error importing package {package_name}: {str(e)}")
            raise
