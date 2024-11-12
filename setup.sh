#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up numerical encoding development environment...${NC}"

# Check if brew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew is not installed. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo -e "${BLUE}Installing Python 3.10...${NC}"
    brew install python@3.10
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3.10 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Install development tools
echo -e "${BLUE}Installing development tools...${NC}"
pip install black isort mypy pytest

echo -e "${GREEN}Setup complete! 🎉${NC}"
echo -e "${GREEN}To activate the environment, run: source venv/bin/activate${NC}"
