# Hotel Booking Cancellation — Pipeline, Grid Search & Gradio Deployment

## Files
- `Hotel_Booking_Pipeline_GridSearch.ipynb` — your original notebook with a new
  section appended at the end: reloads clean data, builds an `sklearn`
  `Pipeline` (preprocessing + XGBoost), tunes it with `GridSearchCV`,
  evaluates it, and saves it as `hotel_cancellation_pipeline.pkl`.
- `app.py` — Gradio web app that loads the `.pkl` and serves predictions.
- `requirements.txt` — Python packages needed to run `app.py`.

## Step 1 — Generate the model file
1. Open `Hotel_Booking_Pipeline_GridSearch.ipynb` in Jupyter (the same way you
   opened the original one).
2. Run all cells top to bottom, including the new **"Production Pipeline +
   Grid Search (XGBoost) + Deployment prep"** section at the end.
   - Grid search runs 24 parameter combinations × 3-fold CV = 72 fits. It may
     take a few minutes depending on your machine.
3. When it finishes, a file named `hotel_cancellation_pipeline.pkl` will be
   saved in the same folder as the notebook.

## Step 2 — Set up the environment (Anaconda Prompt)
Put `app.py`, `requirements.txt`, and `hotel_cancellation_pipeline.pkl` in the
same folder, then open **Anaconda Prompt** and run:

```bat
cd path\to\your\folder

conda create -n hotel-app python=3.10 -y
conda activate hotel-app

pip install -r requirements.txt
```

## Step 3 — Run the app

```bat
python app.py
```

Gradio will print a local URL, typically:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that link in your browser, fill in the form, and click **Predict**.

## Notes
- The Gradio form only asks for the 12 most relevant features (lead time,
  stay length, adults, booking changes, ADR, special requests, parking
  spaces, previous non-canceled bookings, deposit type, customer type,
  market segment) instead of all ~25 columns, to keep it quick to use.
- The `.pkl` contains the *entire* pipeline (scaling + one-hot encoding +
  XGBoost model), so `app.py` never needs to re-implement any preprocessing
  — it just calls `model.predict()` on a raw one-row DataFrame.
- If you want to expose the app on your local network, change the last line
  of `app.py` to `demo.launch(server_name="0.0.0.0")`.
