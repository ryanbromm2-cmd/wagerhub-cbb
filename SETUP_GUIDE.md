# WagerHub Setup Guide
### From zero to a live URL in about 20 minutes.

No coding required. Just clicking through four free websites.

---

## What you'll end up with

- A private URL like `https://wagerhub-cbb.streamlit.app`
- The dashboard refreshes itself every 5 minutes
- New projections and odds are pulled automatically at 9 AM, 12 PM, and 5 PM every day
- You never need to open a terminal again

---

## The four free services you need

| Service | What it does | Cost |
|---|---|---|
| **GitHub** | Stores your code and runs the daily automation | Free |
| **Supabase** | Stores all the data (database) | Free |
| **The Odds API** | Provides live betting lines | Free (500 requests/month) |
| **Streamlit Community Cloud** | Hosts the webpage you visit | Free |

---

## Step 1 — Get a GitHub account

1. Go to **github.com** and click **Sign up**
2. Create a free account
3. After signing in, click the **+** button in the top-right corner → **New repository**
4. Name it `wagerhub-cbb`
5. Set it to **Private**
6. Click **Create repository**

### Upload your files
1. On the new repo page, click **uploading an existing file**
2. Open your `cbb-totals` folder on your computer
3. Select **all files and folders** inside it (Ctrl+A to select all)
4. Drag them into the GitHub upload window
5. Scroll down and click **Commit changes**

---

## Step 2 — Get a free database (Supabase)

1. Go to **supabase.com** and click **Start your project**
2. Sign up with your GitHub account (easiest)
3. Click **New project**
4. Give it any name (e.g. `wagerhub`)
5. Set a database password — write it down somewhere safe
6. Choose any region close to you
7. Click **Create new project** — wait about 2 minutes for it to set up

### Copy your connection string
1. In your Supabase project, click **Settings** (gear icon, bottom left)
2. Click **Database**
3. Scroll down to **Connection string**
4. Click the **URI** tab
5. Copy the string — it looks like:
   `postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres`
6. **Replace `[YOUR-PASSWORD]`** in the string with the password you set in step 5
7. Save this string — you'll use it in Steps 3 and 4

---

## Step 3 — Get your Odds API key

1. Go to **the-odds-api.com**
2. Click **Get API Key** (free — no credit card needed)
3. Sign up and copy your API key
4. The free tier gives you 500 requests/month which is plenty

---

## Step 4 — Add your secrets to GitHub

This tells the daily automation your API keys without putting them in the code.

1. Go to your `wagerhub-cbb` repo on github.com
2. Click **Settings** (top tab row)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret** and add these one at a time:

| Name | Value |
|---|---|
| `DATABASE_URL` | The full PostgreSQL string from Step 2 |
| `THE_ODDS_API_KEY` | Your key from Step 3 |

5. That's it for GitHub

---

## Step 5 — Set up the live dashboard (Streamlit)

1. Go to **share.streamlit.io**
2. Click **Sign in with GitHub** and authorize it
3. Click **New app**
4. Fill in:
   - **Repository:** `your-github-username/wagerhub-cbb`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
5. Click **Advanced settings** → **Secrets**
6. Paste this into the secrets box (replace the values with your actual keys):

```toml
THE_ODDS_API_KEY = "paste-your-key-here"
DATABASE_URL = "paste-your-supabase-connection-string-here"
```

7. Click **Save** then click **Deploy**
8. Wait 2–3 minutes while it builds
9. You'll get a URL — **bookmark this, it's your WagerHub dashboard**

---

## Step 6 — Run the pipeline for the first time

1. Go to your `wagerhub-cbb` repo on github.com
2. Click the **Actions** tab
3. Click **CBB Totals — Daily Pipeline** in the left list
4. Click **Run workflow** → **Run workflow**
5. Wait about 2–3 minutes
6. Open your Streamlit URL — you should see real data

---

## You're done

After this one-time setup:

- **Every day**, the system automatically runs at 9 AM, 12 PM, and 5 PM ET
- **The dashboard** refreshes itself every 5 minutes
- **You just open the URL** and see the slate

### Your daily routine
1. Open your bookmarked URL in the morning
2. Look at today's slate — biggest edges are at the top
3. That's it

---

## Manual refresh (optional)

If you want new odds pulled right now (outside the scheduled times):
1. Go to your GitHub repo → **Actions** tab
2. Click **CBB Totals — Daily Pipeline**
3. Click **Run workflow** → choose **refresh-odds** → **Run workflow**
4. Wait 2 minutes, refresh your dashboard

---

## Privacy

Your Streamlit app URL is technically public by default on the free tier.
To make it private (invite-only):
1. In Streamlit Cloud, click your app → **Settings** → **Sharing**
2. Change to **Only specific people can view this app**
3. Add email addresses of people you want to allow

---

## Troubleshooting

**Dashboard shows "Demo Mode"**
→ The pipeline hasn't run yet, or the DATABASE_URL secret is wrong.
→ Go to GitHub Actions and run the pipeline manually (Step 6 above).

**GitHub Actions failing**
→ Click the failed run to see the error log.
→ Most common cause: DATABASE_URL secret is wrong (re-check the password replacement).

**Streamlit app not loading**
→ Free apps go to sleep after 7 days of no visits — just open the URL and wait 30 seconds for it to wake up.

**Odds not showing**
→ Check your THE_ODDS_API_KEY secret. The free tier is 500 requests/month — check usage at the-odds-api.com.
