# EDXtoolkit
**A Bachelor Thesis Project by Linh Zimmermann (University of Bonn, August 2025)**

## Overview  
Welcome, friends!!  

This program is an interactive EDX spectrum analysis tool built with Flask + Dash. It allows you to:  
- Upload vendor-specific CSV files (EDX Tescan Clara) containing Energy-Dispersive X-ray (EDX) spectra  
- Smooth and background-correct the spectrum  
- Automatically fit peaks with a pseudo-Voigt profile  
- Match peaks against reference line-energy libraries (NIST + optional PyXRay)  
- Assign or remove line candidates interactively  
- Export assigned candidates as CSV  

The program runs in your web browser locally and provides an intuitive GUI, even if you have never programmed in Python.

## Requirements  

Below is a list of required Python packages (as specified in `requirements.txt`):

```text
# --- Core web framework ---
flask>=2.3

# --- Dash UI (runs on Flask) ---
dash>=2.14
plotly>=5.20

# --- Data & numerics ---
pandas>=2.1  
numpy>=1.24  
scipy>=1.10

# --- Optional: X-ray reference lines (uncomment if you want it) ---
# pyxray>=0.5.10
```

- **Flask** – backend web framework  
- **Dash** and **Plotly** – interactive GUI and plots  
- **Pandas** and **NumPy** – data handling and computation  
- **SciPy** – scientific routines (filtering, curve fitting, peak detection)  
- **PyXRay** – optional: additional reference line library  

To install all packages in one step, run:  
```bash
pip install -r requirements.txt
```

## Installation Instructions  

Follow the guide for your operating system :

### Windows 10/11  
1. Download and install Python 3 from [python.org](https://www.python.org) (make sure to check **“Add Python to PATH”** during installation).  
2. Verify the installation by running `python --version` in Command Prompt.  
3. Navigate to your project folder in Command Prompt and install the required packages:  
   ```bash
   py -m pip install -r requirements.txt
   ```

### macOS  
1. Open the Terminal app and check if Python 3 is installed by running `python3 --version`. (If not found, download and install Python 3 from [python.org](https://www.python.org).)  
2. Install the required packages by running:  
   ```bash
   python3 -m pip install -r requirements.txt
   ```

### Linux  
1. Most Linux distributions include Python 3. Verify by running `python3 --version`. (If missing, install Python via your package manager. For example, on Debian/Ubuntu: `sudo apt install python3 python3-pip`.)  
2. Install the required packages:  
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Configuring the CSV Path  

In the Python script (`EDXtoolkit.py`), locate the section near the top that looks like:  
```python
# Path to your EDX CSV spectrum file
CSV_PATH = "/path/to/your/data.csv"
```  
Update this line to point to the sample spectrum CSV file (found in the folder *samples and libraries*):  
- **Windows:** Copy the full file path of your CSV file (Shift + Right Click → “Copy as path”), then set the `CSV_PATH` variable. For example:  
  ```python
  CSV_PATH = r"C:\Users\YourName\Documents\EDX\data.csv"
  ```  
  *(Use the `r"..."` raw string format or double backslashes `\\` for Windows paths.)*  
- **macOS/Linux:** Use forward slashes in the file path. For example:  
  ```python
  CSV_PATH = "/home/username/data/data.csv"
  ```
  Repeat the same for the CSV file `nist_all_energies.csv`which is also found in the folder *samples and libraries*
  
## Running the Program  

**Option A – Visual Studio Code:**  
1. Open the project folder (with the Python script) in VS Code.  
2. Install the **Python** extension in VS Code (if not already installed).  
3. Run the script by clicking **“Run Python File in Terminal”** (the green play button at the top-right of the editor).  
4. Open your web browser at the internal URL printed in the Terminal.  

**Option B – Command Line:**  
1. Open a terminal/command prompt and navigate to the script’s folder.  
2. Run the program with `EDXtoolkit.py` (use `python3 EDXtoolkit.py` on macOS/Linux).  
3. Open your web browser at the internal URL printed in the Terminal.  

## Usage  

- **Upload Spectrum:** Drag & drop your CSV file into the upload box.  
- **Preprocessing:** Use sliders to smooth the data, set peak detection thresholds, and toggle background subtraction.  
- **Peak Detection:** Peaks are automatically fitted with pseudo-Voigt profiles.  
- **Line Matching:** Detected peaks are compared against known X-ray line energies (NIST + optional PyXRay).  
- **Manual Assignment:** Click on a detected peak to view candidate element lines, then assign/remove them.  
- **Pile-up Handling:** Optional detection and marking of pile-up peaks.  
- **Export Results:** Save your assigned peak data to a CSV file.  

> **Note:** Do not edit the code except for the `CSV_PATH`. All analysis is controlled via the web interface.
