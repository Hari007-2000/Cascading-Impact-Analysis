# Deploying the Cascading Impact app to Streamlit Community Cloud

This gets you a public `https://<your-name>.streamlit.app/` URL, like the
example you shared. It's free. You do the deploy from your own Streamlit
account (it signs in with GitHub) — it can't be done on your behalf.

## Step 1 — Put these 4 files in your GitHub repo (root level)

Your repo `Hari007-2000/Cascading-Impact-Analysis` needs these files at the
top level (the CSV is already there):

- `app.py`
- `model.py`
- `requirements.txt`
- `PIOT_ModelD_APAP_workshop.csv`   ← already in your repo

**Easiest way (no command line):**
1. Go to https://github.com/Hari007-2000/Cascading-Impact-Analysis
2. Click **Add file → Upload files**
3. Drag in `app.py`, `model.py`, and `requirements.txt`
4. Click **Commit changes**

**Or with git:**
```bash
git clone https://github.com/Hari007-2000/Cascading-Impact-Analysis.git
cd Cascading-Impact-Analysis
cp /path/to/app.py /path/to/model.py /path/to/requirements.txt .
git add app.py model.py requirements.txt
git commit -m "Add Streamlit cascading-impact app"
git push
```

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account
   (authorize it to see your repositories the first time).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `Hari007-2000/Cascading-Impact-Analysis`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** pick a custom subdomain, e.g. `apap-cascade`
     → gives `https://apap-cascade.streamlit.app`
4. Click **Deploy**. First build takes ~1–2 minutes while it installs
   `requirements.txt`.

That's it — the URL is live and shareable. Every time you `git push` to
`main`, the app redeploys automatically.

## Notes

- The repo can stay **public** (required for the free tier) or, on some plans,
  private. The app itself is publicly reachable by anyone with the URL.
- No secrets or API keys are needed — everything runs from the bundled CSV.
- To change the baseline data, either replace the CSV in the repo or use the
  **Upload a custom PIOT CSV** control in the app's sidebar.
- If the build fails, open **Manage app → logs** on Streamlit Cloud; the most
  common cause is a missing package in `requirements.txt`.
