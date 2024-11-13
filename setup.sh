#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up numerical encoding environment...${NC}"

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies
echo -e "${YELLOW}Installing development dependencies...${NC}"
pip install -e ".[dev]"

# Set up pre-commit hooks
if [ -f ".git/hooks/pre-commit" ]; then
    echo -e "${YELLOW}Setting up pre-commit hooks...${NC}"
    # Add your pre-commit setup here if needed
    echo "#!/bin/bash
    black . --check
    isort . --check
    mypy numerical_encoding" > .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
fi

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "To activate the virtual environment, run: ${YELLOW}source venv/bin/activate${NC}"