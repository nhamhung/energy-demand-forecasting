# Start here: run the Energy Demand Forecasting notebook and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code, a Kaggle account or a Streamlit account. You double-click one file, and it
sets everything up.

**What you get**

- **The notebook**: a step-by-step walk-through of how the model forecasting Romania's hourly
  electricity demand is built (the data and its quirks, every feature, the models, 24-hour
  forecasts, and what the model relies on), that you can run, change and re-run in your web
  browser.
- **The app**: the interactive Energy Demand Forecasting app (predict any hour or forecast up to
  24 hours ahead, explore the data and the model) running on your own computer.

The data comes with the project: every hour from January 2019 to 14 March 2026 (the notebook
uses all of it; the app starts on the last year to be quick, and can use all of it too). Getting
newer data from Kaggle is optional and needs a free Kaggle account: see
[Get the newest data from Kaggle](#optional-get-the-newest-data-from-kaggle).

**What you need**

- Windows 10 or 11, or macOS 12 or newer (Apple silicon or Intel).
- An internet connection the first time (it downloads a few hundred MB).
- About 2 GB of free disk space.

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/energy-demand-forecasting>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `energy-demand-forecasting-main.zip` and choose
   **Extract All...**. Extract it to a simple place such as
   `C:\Users\<you>\Documents\energy-demand-forecasting-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Energy Demand Forecasting ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Open the app on the full dataset (2019-2026, slower to start)
     4  Quit
   Type 1, 2, 3 or 4 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/energy-demand-forecasting>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `energy-demand-forecasting-main.zip` to unzip
   it. Move the `energy-demand-forecasting-main` folder somewhere simple, such as your
   **Documents** folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
   On most Macs the first start also prints *"One-time fix: downloading the
   OpenMP runtime..."*: that is a small library (about 300 kB) the model needs on macOS, and it is
   stored inside the project folder.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks), **2** (app) or
   **3** (app on the full dataset) and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebook (choice 1)

JupyterLab opens in your web browser.

1. In the file list on the left, double-click **`01_eda_and_modeling.ipynb`**.
2. Run it from top to bottom with **Run → Run All Cells**. The whole notebook takes about 3–6
   minutes depending on your computer. A `[*]` next to a cell means it is still running.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a number in a cell and run it again. The notebook ends with
   suggested exercises.
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

Good to know:

- The notebook uses the full dataset when you have downloaded it; otherwise it uses the bundled
  one-year sample, so some numbers will differ from the text.
- The notebook never downloads anything and never changes the project's data or model files.

Your edits are saved in the notebook file. To keep the original, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2 or 3)

**Choice 2** runs the app on the last year of data (March 2025 to March 2026), so it starts
quickly. **Choice 3** runs it on every hour since 2019 (the first page load takes a bit longer).

The app opens in your browser by itself (if it doesn't, copy the `http://localhost:...` address
shown in the black/Terminal window into your browser). Pages, in the menu on the left:

- **Predict**: pick any hour and compare the model's prediction with what really happened, or
  forecast up to 24 hours ahead and see how the error grows with each hour.
- **Dataset Overview**: the demand over time, the generation mix and the renewable share by year.
- **Feature Engineering**: the inputs the model uses and the statistical checks behind them.
- **Model Insights**: the model compared with simple baselines, and what it relies on (SHAP).

The first time a page loads it may take up to a minute (it prepares the data and model). Nothing
you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup has finished.

---

## Optional: get the newest data from Kaggle

You don't need this: the project already includes the dataset up to 14 March 2026. The dataset
is published on
[Kaggle](https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production)
and its author adds new hours from time to time. To download the newest version you need a free
Kaggle account and an *API token* (a password-like code that lets the project download from
Kaggle for you). You set this up once.

> With newer data, the notebook's numbers drift away from its text, because the model is then
> tested on more recent hours. Keep a copy of the original file if you want to go back (step 3).

### 1. Create a Kaggle account and an API token (same on Windows and macOS)

1. Go to <https://www.kaggle.com> and click **Register**. Sign up with Google or with your email
   address, and confirm your email if Kaggle asks you to.
2. When you're signed in, open <https://www.kaggle.com/settings/api> (or click your profile
   picture → **Settings** and scroll to **API**).
3. Click **Generate New Token**, give it any name if asked, and **copy the token** it shows.
   Keep this window open until step 4 is done: treat the token like a password.

### 2. Close the project window

If the black/Terminal window is open, stop it (see *Stop it* above).

### 3. Move the included data file out of the way

In the project folder, open `data`, then `raw`, and **rename** `Romania_hourly.csv` to
`Romania_hourly_2026-03-14.csv` (or move it somewhere else). The project downloads a new copy
when this file is missing.

- **Windows:** in File Explorer, right-click the file → **Rename**. (If Explorer shows the
  name as just `Romania_hourly`, type `Romania_hourly_2026-03-14`: the hidden `.csv` stays.)
- **macOS:** in Finder, click the file once, press **Return**, type the new name, press
  **Return**.

To go back to the original later: delete the new `Romania_hourly.csv` and rename your copy back.

### 4. Download the newest data

1. Double-click the start file as usual and, in the menu, type **3**
   (*Open the app on the full dataset*) and press Enter.
2. It asks *"Paste your Kaggle API token"*. Paste the token and press Enter.
   - **Windows:** right-click inside the black window (or press **Ctrl + V**) to paste.
   - **macOS:** press **Command + V**.
3. It saves the token on your computer (you won't be asked again), downloads the newest dataset
   (a few seconds) to `data/raw/Romania_hourly.csv`, and opens the app on it. The notebook
   (choice 1) uses the new file too.

**If you prefer to save the token yourself** (instead of pasting it into the window): save the
token as a text file named `access_token` (no file extension) in a folder named `.kaggle` in
your home folder:

- **Windows:** `C:\Users\<you>\.kaggle\access_token`. In Notepad, choose **Save as type: All
  files** so it doesn't become `access_token.txt` (that name also works).
- **macOS:** `/Users/<you>/.kaggle/access_token`. The `.kaggle` folder is hidden in Finder;
  press **Command + Shift + .** to show hidden folders.

An older `kaggle.json` key file from Kaggle also works when saved in that same `.kaggle` folder.

**If the download fails**: the token is usually mistyped or expired. Generate a new one on
<https://www.kaggle.com/settings/api>, delete the `access_token` file from the `.kaggle` folder
above, and choose 3 again. Until then, put your original file back (step 3) to keep everything
working.

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebook:

1. Run the start file once (above), so the project's Python environment (the `.venv` folder)
   exists. You can quit from the menu.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open `notebooks/01_eda_and_modeling.ipynb`. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once.

On a Mac, VS Code doesn't get the launcher's OpenMP fix: if a cell fails with *"Library not
loaded: @rpath/libomp.dylib"*, either use the launcher's JupyterLab instead, or install
[Homebrew](https://brew.sh) and run `brew install libomp` in Terminal once.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| macOS: *"Library not loaded: @rpath/libomp.dylib"* or *"brew install libomp"* | The launcher normally fixes this by itself (it needs the internet once). If it says the fix didn't work, install [Homebrew](https://brew.sh), run `brew install libomp` in Terminal, and start again. |
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| The start file asks to download the dataset, or the notebook shows errors about dates | The optional file `data/raw/Romania_hourly.csv` is missing. Download it using [Get the newest data from Kaggle](#optional-get-the-newest-data-from-kaggle), or choose the bundled-sample option. |
| The download from Kaggle didn't work | See *If the download fails* in [Get the newest data from Kaggle](#optional-get-the-newest-data-from-kaggle). Choice 2 always works without Kaggle. |
| The notebook says the model file is missing | `models/model.joblib` ships with the project. Download the ZIP again for a complete copy. |
| Errors after editing the notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy. |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup. |

## Uninstall

1. Delete the project folder.
2. Optional, to remove uv and the downloaded Python too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folder
     `%APPDATA%\uv`, and the folder `%LOCALAPPDATA%\uv`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv`.
3. Optional, if you set up Kaggle: delete the `.kaggle` folder in your home folder
   (Windows: `C:\Users\<you>\.kaggle`; macOS: run `rm -rf ~/.kaggle` in Terminal), and delete
   the token on <https://www.kaggle.com/settings/api>.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.
(One exception: Intel Macs get numpy 2.3.5 instead of 2.5.3, because shap needs numba and the
newest numba with a ready-built Intel-Mac wheel supports numpy only up to 2.3.)

```bash
uv sync                                   # create .venv (add --no-dev to skip pytest/ruff)
uv run pytest                             # tests
uv run jupyter lab notebooks              # notebooks
uv run streamlit run app/streamlit_app.py # app
```

The bundled sample in `data/sample/` runs without Kaggle. The optional full
`data/raw/Romania_hourly.csv` snapshot is downloaded through the Kaggle API when requested
(`USE_FULL_KAGGLE_DATA=true`).
`scripts/launch.py` is what the start files run. `requirements.txt` and `app/requirements.txt`
stay for Streamlit Community Cloud and Docker. To publish your own changes to GitHub, see the
setup manual at <https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS
Code, clone, push).
