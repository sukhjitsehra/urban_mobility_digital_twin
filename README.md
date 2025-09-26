# Urban Mobility Digital Twin - DTUMOS-Disabled-CallTaxi

This project is part of the **Urban Mobility Digital Twin (DTUMOS)** initiative, focusing on simulating and analyzing urban mobility systems.  
The `DTUMOS-Disabled-CallTaxi` branch specifically addresses scenarios where **call taxi services are disabled**.

---

## 🚀 Prerequisites

Before running the project, make sure you have:

- Python 3.8 or higher  
- WSL (for Windows) or Linux  
- Git  
- Docker (for running OSRM backend)  
- Jupyter Notebook  

---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/sukhjitsehra/urban_mobility_digital_twin.git
cd urban_mobility_digital_twin
git checkout DTUMOS-Disabled-CallTaxi
2. Create and activate a virtual environment
Windows (with WSL):

bash
Copy code
python -m venv venv
source venv/Scripts/activate
Linux/macOS:

bash
Copy code
python -m venv venv
source venv/bin/activate
3. Install required Python packages
bash
Copy code
pip install -r requirements.txt
If requirements.txt is not present, manually install packages as needed.

🗺️ OSRM Backend Setup
The OSRM dataset is already preprocessed and stored in the Hugging Face repository: kaur6910/osrm.

1. Download the Dataset
Install Git LFS if not already installed, then clone the dataset:

bash
Copy code
git lfs install
git clone https://huggingface.co/datasets/kaur6910/osrm
This will create a folder named osrm containing all the preprocessed OSRM files.

2. Move the Dataset to the DTUMOS Data Folder
bash
Copy code
mkdir -p thirdparty/dtumos/data
mv osrm thirdparty/dtumos/data/osrm
After this step, your project structure should look like:

ruby
Copy code
urban_mobility_digital_twin/
│
├── thirdparty/
│   └── dtumos/
│       └── data/
│           └── osrm/         # Preprocessed OSRM files from Hugging Face
├── main.ipynb
├── module/
└── README.md
3. Run the OSRM Backend
Start the OSRM routing engine:

bash
Copy code
docker pull osrm/osrm-backend
docker run -t -i -p 5000:5000 \
    -v $(pwd)/thirdparty/dtumos/data/osrm:/data \
    osrm/osrm-backend osrm-routed --algorithm mld /data/berlin-latest.osrm
The OSRM backend will now be accessible at http://localhost:5000.

🛠️ Running the Project
Ensure your virtual environment is activated.

Ensure the OSRM backend is running.

Launch Jupyter Notebook:

bash
Copy code
jupyter notebook
Open main.ipynb in your browser.

Run the notebook cells sequentially to execute the simulation and analysis.

All simulations and analyses are executed through the notebook, which connects to the OSRM backend and uses the DTUMOS framework.
