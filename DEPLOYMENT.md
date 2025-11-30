# Deployment Guide for DermAI

This guide explains how to deploy the DermAI application to **Render.com**.

## Prerequisites

1.  **GitHub Account**: You need a GitHub account to host your code.
2.  **Render Account**: Sign up at [render.com](https://render.com/).

## Step 1: Push Code to GitHub

1.  Initialize a git repository (if not already done):
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    ```
2.  Create a new repository on GitHub.
3.  Link your local folder to the GitHub repo and push:
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    git branch -M main
    git push -u origin main
    ```

## Step 2: Deploy on Render

1.  Log in to your **Render Dashboard**.
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub account and select the `derm_ai` repository.
4.  Configure the service:
    *   **Name**: `derm-ai` (or any unique name)
    *   **Region**: Choose the one closest to you.
    *   **Branch**: `main`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn site_server:app`
    *   **Instance Type**: `Free`
5.  Click **Create Web Service**.

## Step 3: Wait for Build

Render will now clone your repo, install dependencies, and start the server. This process may take a few minutes.
Once finished, you will see a green "Live" badge and a URL (e.g., `https://derm-ai.onrender.com`).

## Important Notes

*   **Data Reset**: On the free tier, your database (`users.db`) and uploaded images will be wiped whenever the server restarts or "spins down" due to inactivity.
*   **Spin Down**: The free tier spins down after 15 minutes of inactivity. The first request after that will take about 30-60 seconds to load.
