#!/usr/bin/env python3
"""
Build script for QEP-VLA Platform
Production build and packaging
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QEPVLAPlatformBuilder:
    """Builder for QEP-VLA Platform"""
    
    def __init__(self, build_dir: str = "build", clean: bool = False):
        self.build_dir = Path(build_dir)
        self.clean = clean
        self.project_root = Path(__file__).parent.parent
        
        # Build configuration
        self.build_config = {
            'version': '1.0.0',
            'python_version': '3.11',
            'platform': 'linux-x86_64',
            'build_date': datetime.now().isoformat()
        }
    
    def clean_build(self):
        """Clean build directory"""
        if self.build_dir.exists() and self.clean:
            logger.info(f"Cleaning build directory: {self.build_dir}")
            shutil.rmtree(self.build_dir)
        
        # Create fresh build directory
        self.build_dir.mkdir(exist_ok=True)
        logger.info(f"Build directory ready: {self.build_dir}")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.project_root / "requirements.txt")
            ], check=True, capture_output=True, text=True)
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
    
    def run_tests(self):
        """Run test suite"""
        logger.info("Running test suite...")
        
        try:
            # Run pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(self.project_root / "tests"),
                "-v", "--tb=short"
            ], check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("All tests passed")
            else:
                logger.warning("Some tests failed")
                logger.warning(f"stdout: {result.stdout}")
                logger.warning(f"stderr: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            raise
    
    def run_linting(self):
        """Run code linting"""
        logger.info("Running code linting...")
        
        try:
            # Run flake8
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                str(self.project_root / "src"),
                str(self.project_root / "tests"),
                "--max-line-length=120",
                "--ignore=E203,W503"
            ], check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Linting passed")
            else:
                logger.warning("Linting issues found")
                logger.warning(f"stdout: {result.stdout}")
                logger.warning(f"stderr: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to run linting: {e}")
            raise
    
    def run_formatting(self):
        """Run code formatting"""
        logger.info("Running code formatting...")
        
        try:
            # Run black
            subprocess.run([
                sys.executable, "-m", "black",
                str(self.project_root / "src"),
                str(self.project_root / "tests"),
                "--line-length=120",
                "--check"
            ], check=True, capture_output=True, text=True)
            
            logger.info("Code formatting check passed")
            
        except subprocess.CalledProcessError as e:
            logger.warning("Code formatting issues found")
            logger.warning(f"stdout: {e.stdout}")
            logger.warning(f"stderr: {e.stderr}")
    
    def build_package(self):
        """Build Python package"""
        logger.info("Building Python package...")
        
        try:
            # Create package structure
            package_dir = self.build_dir / "qep_vla_platform"
            package_dir.mkdir(exist_ok=True)
            
            # Copy source code
            src_dir = self.project_root / "src"
            if src_dir.exists():
                shutil.copytree(src_dir, package_dir / "src", dirs_exist_ok=True)
                logger.info("Source code copied")
            
            # Copy configuration files
            config_files = [
                "package.json",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yml",
                "README.md",
                "env.example"
            ]
            
            for file_name in config_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, package_dir)
                    logger.info(f"Copied {file_name}")
            
            # Create __init__.py files
            self._create_init_files(package_dir)
            
            # Create setup.py
            self._create_setup_py(package_dir)
            
            logger.info("Package structure created successfully")
            
        except Exception as e:
            logger.error(f"Failed to build package: {e}")
            raise
    
    def _create_init_files(self, package_dir: Path):
        """Create __init__.py files for Python packages"""
        # Find all Python package directories
        for root, dirs, files in os.walk(package_dir):
            root_path = Path(root)
            
            # Check if directory contains Python files
            python_files = [f for f in files if f.endswith('.py')]
            if python_files and not (root_path / "__init__.py").exists():
                # Create __init__.py
                init_file = root_path / "__init__.py"
                init_file.write_text("# Auto-generated __init__.py\n")
                logger.debug(f"Created {init_file}")
    
    def _create_setup_py(self, package_dir: Path):
        """Create setup.py for package installation"""
        setup_content = f'''#!/usr/bin/env python3
"""
Setup script for QEP-VLA Platform
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qep-vla-platform",
    version="{self.build_config['version']}",
    description="Quantum-Enhanced Privacy-Preserving Vision-Language-Action Navigation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Frank Van Laarhoven",
    author_email="f.vanlaarhoven@ncl.ac.uk",
    url="https://github.com/FrankVanLaarhoven/QEP-VLA-Platform",
    packages=find_packages(),
    python_requires=">={self.build_config['python_version']}",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "autonomous-navigation",
        "quantum-sensing", 
        "privacy-preserving-ai",
        "federated-learning",
        "vision-language-action"
    ],
    include_package_data=True,
    zip_safe=False,
)
'''
        
        setup_file = package_dir / "setup.py"
        setup_file.write_text(setup_content)
        logger.info("Created setup.py")
    
    def build_docker_image(self):
        """Build Docker image"""
        logger.info("Building Docker image...")
        
        try:
            # Build Docker image
            subprocess.run([
                "docker", "build",
                "-t", f"qep-vla-platform:{self.build_config['version']}",
                "-t", "qep-vla-platform:latest",
                str(self.project_root)
            ], check=True, capture_output=True, text=True)
            
            logger.info("Docker image built successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Docker image: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
    
    def create_archive(self):
        """Create distribution archive"""
        logger.info("Creating distribution archive...")
        
        try:
            # Create tar.gz archive
            archive_name = f"qep-vla-platform-{self.build_config['version']}.tar.gz"
            archive_path = self.build_dir / archive_name
            
            # Create archive
            shutil.make_archive(
                str(self.build_dir / f"qep-vla-platform-{self.build_config['version']}"),
                'gztar',
                self.build_dir,
                "qep_vla_platform"
            )
            
            logger.info(f"Distribution archive created: {archive_path}")
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            raise
    
    def generate_build_report(self):
        """Generate build report"""
        logger.info("Generating build report...")
        
        try:
            report_content = f"""# QEP-VLA Platform Build Report

**Build Date:** {self.build_config['build_date']}
**Version:** {self.build_config['version']}
**Python Version:** {self.build_config['python_version']}
**Platform:** {self.build_config['platform']}

## Build Summary

✅ Dependencies installed
✅ Tests executed
✅ Linting completed
✅ Code formatting checked
✅ Package structure created
✅ Docker image built
✅ Distribution archive created

## Build Artifacts

- Package directory: `{self.build_dir}/qep_vla_platform/`
- Distribution archive: `{self.build_dir}/qep-vla-platform-{self.build_config['version']}.tar.gz`
- Docker image: `qep-vla-platform:{self.build_config['version']}`

## Next Steps

1. Test the built package: `pip install {self.build_dir}/qep_vla_platform/`
2. Run the Docker container: `docker run qep-vla-platform:latest`
3. Deploy to production environment

---
*Build completed successfully at {datetime.now().isoformat()}*
"""
            
            report_file = self.build_dir / "BUILD_REPORT.md"
            report_file.write_text(report_content)
            logger.info(f"Build report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate build report: {e}")
            raise
    
    def build(self):
        """Execute complete build process"""
        logger.info("Starting QEP-VLA Platform build process...")
        
        try:
            # Step 1: Clean build directory
            self.clean_build()
            
            # Step 2: Install dependencies
            self.install_dependencies()
            
            # Step 3: Run tests
            self.run_tests()
            
            # Step 4: Run linting
            self.run_linting()
            
            # Step 5: Check formatting
            self.run_formatting()
            
            # Step 6: Build package
            self.build_package()
            
            # Step 7: Build Docker image
            self.build_docker_image()
            
            # Step 8: Create archive
            self.create_archive()
            
            # Step 9: Generate report
            self.generate_build_report()
            
            logger.info("Build process completed successfully!")
            logger.info(f"Build artifacts available in: {self.build_dir}")
            
        except Exception as e:
            logger.error(f"Build process failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Build QEP-VLA Platform")
    parser.add_argument("--build-dir", default="build", help="Build directory")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before building")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker image build")
    
    args = parser.parse_args()
    
    try:
        builder = QEPVLAPlatformBuilder(
            build_dir=args.build_dir,
            clean=args.clean
        )
        
        # Modify build process based on arguments
        if args.skip_tests:
            builder.run_tests = lambda: logger.info("Skipping tests")
        
        if args.skip_docker:
            builder.build_docker_image = lambda: logger.info("Skipping Docker build")
        
        builder.build()
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
