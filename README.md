# pc5253-traffic-simulation
Final project for PC 5253

## Setup Instructions

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Prerequisites

Install uv using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**With pip:**
```bash
pip install uv
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pc5253-traffic-simulation.git
cd pc5253-traffic-simulation
```

2. Sync dependencies and create virtual environment:
```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies specified in the project.

### Setting Up Jupyter Kernel

To use this project's virtual environment in Jupyter notebooks:

1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install ipykernel if not already installed:
```bash
uv pip install ipykernel
```

3. Create a Jupyter kernel for this project:
```bash
python -m ipykernel install --user --name=pc5253-traffic --display-name "Python (pc5253-traffic)"
```

4. Now you can select the "Python (pc5253-traffic)" kernel in Jupyter notebooks or JupyterLab.

To remove the kernel later:
```bash
jupyter kernelspec uninstall pc5253-traffic
```
