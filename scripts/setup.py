#!/usr/bin/env python3
"""
Setup script for ConsciousnessX
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_dependencies():
    """Check for required dependencies"""
    required = ["docker", "docker-compose", "python3", "pip"]
    missing = []
    
    for cmd in required:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them before continuing.")
        sys.exit(1)
    
    print("✓ All dependencies found")


def setup_environment():
    """Setup environment"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✓ Created .env file")
        print("Please edit .env file with your configuration")
    else:
        print("✓ .env file already exists")
    
    # Create required directories
    directories = [
        "uploads",
        "logs",
        "data",
        "monitoring",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✓ Created required directories")


def setup_database():
    """Initialize database"""
    print("Setting up database...")
    
    # Create init SQL
    init_sql = """
    -- Initialize ConsciousnessX database
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    
    -- Create additional tables and indexes here
    """
    
    with open("init-db.sql", "w") as f:
        f.write(init_sql)
    
    print("✓ Created database initialization script")


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    requirements = [
        "requirements/prod.txt",
        "requirements/dev.txt",  # if exists
    ]
    
    for req in requirements:
        if Path(req).exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req])
            print(f"✓ Installed dependencies from {req}")


def build_docker():
    """Build Docker images"""
    print("Building Docker images...")
    
    result = subprocess.run(
        ["docker-compose", "build"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Docker images built successfully")
    else:
        print("✗ Docker build failed")
        print(result.stderr)
        sys.exit(1)


def print_next_steps():
    """Print next steps for setup"""
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Edit the .env file with your API keys and configuration")
    print("2. Start the services: docker-compose up -d")
    print("3. Access the API: http://localhost:8000")
    print("4. Access the WebUI: http://localhost:8501")
    print("5. View metrics: http://localhost:9090 (Prometheus)")
    print("6. View dashboards: http://localhost:3000 (Grafana)")
    print("\nFor production deployment:")
    print("1. Configure SSL certificates")
    print("2. Set up proper secrets management")
    print("3. Configure backup strategies")
    print("4. Set up monitoring alerts")
    print("\nDocumentation: https://github.com/Napiersnotes/consciousnessX")
    print("="*50)


def main():
    """Main setup function"""
    print("🧠 ConsciousnessX Production Setup")
    print("-" * 40)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run setup steps
    check_dependencies()
    setup_environment()
    setup_database()
    install_dependencies()
    build_docker()
    print_next_steps()


if __name__ == "__main__":
    main()
