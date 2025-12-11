# DEPLOYMENT GUIDE
**Doc-Understand - Google Cloud Run Deployment**

---

# OVERVIEW

This document explains how to deploy the Doc-Understand system to Google Cloud Run with automated CI/CD using Cloud Build.

**Deployment Stack:**
- **Container Platform:** Docker
- **Hosting:** Google Cloud Run (serverless containers)
- **CI/CD:** Google Cloud Build (auto-deploy on git push)
- **Storage:** Google Cloud Storage + AWS S3
- **Monitoring:** Google Cloud Monitoring

---

# PREREQUISITES

## 1. Accounts Required

- [ ] Google Cloud Platform account
- [ ] AWS account (for Textract)
- [ ] GitHub account
- [ ] Groq account (for LLM API)

## 2. Local Tools (Optional - can use Cloud Shell)

- [ ] Git installed
- [ ] Docker installed (for local testing)
- [ ] gcloud CLI installed

## 3. Access & Permissions

- [ ] GCP Project created: `doc-understand`
- [ ] Billing enabled on GCP project
- [ ] Owner or Editor role on GCP project

---

# QUICK START (5 Minutes)

If you just want to deploy quickly:

```bash
# 1. Clone repository
git clone https://github.com/your-username/doc-understand.git
cd doc-understand

# 2. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Deploy to Cloud Run
gcloud run deploy doc-understand \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# 4. Get the URL
gcloud run services describe doc-understand --region us-central1 --format 'value(status.url)'
```

**Done!** Your app is live.

---

# DETAILED SETUP (Step-by-Step)

## STEP 1: Enable Google Cloud APIs

**In Google Cloud Console or via CLI:**

```bash
# Set your project
gcloud config set project doc-understand

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable monitoring.googleapis.com
```

**What These APIs Do:**
- **Cloud Run:** Runs your containerized app
- **Cloud Build:** Builds Docker images automatically
- **Container Registry:** Stores Docker images
- **Cloud Storage:** Already using for data
- **Cloud Monitoring:** Already using for metrics

---

## STEP 2: Configure Secrets

**Option A: Using Secret Manager (Recommended)**

```bash
# Create secrets
echo -n "your-groq-api-key" | gcloud secrets create groq-api-key --data-file=-
echo -n "your-aws-access-key" | gcloud secrets create aws-access-key --data-file=-
echo -n "your-aws-secret-key" | gcloud secrets create aws-secret-key --data-file=-

# Upload GCP service account JSON
gcloud secrets create gcp-service-account --data-file=gcp_keys/doc-understand-*.json
```

**Option B: Using Environment Variables (Simpler for Testing)**

```bash
# Set environment variables on Cloud Run service
gcloud run services update doc-understand \
  --set-env-vars GROQ_API_KEY=your-key,AWS_ACCESS_KEY_ID=your-key \
  --region us-central1
```

---

## STEP 3: Test Docker Locally (Optional but Recommended)

**Build the image:**
```bash
docker build -t doc-understand .
```

**Run locally:**
```bash
docker run -p 8080:8080 \
  -e GROQ_API_KEY=your-key \
  -e AWS_ACCESS_KEY_ID=your-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret \
  doc-understand
```

**Test:**
- Open: http://localhost:8080
- Upload sample PDF
- Ask question
- Verify answer appears

**Stop:**
```bash
docker stop $(docker ps -q --filter ancestor=doc-understand)
```

---

## STEP 4: Deploy to Cloud Run (Manual First Time)

**Deploy command:**
```bash
gcloud run deploy doc-understand \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

**What happens:**
1. Cloud Build automatically builds Docker image
2. Pushes to Container Registry
3. Deploys to Cloud Run
4. Returns public URL

**Expected output:**
```
Service [doc-understand] revision [doc-understand-00001-xxx] has been deployed
Service URL: https://doc-understand-xxxxx-uc.a.run.app
```

**Test the deployment:**
- Copy the Service URL
- Open in browser
- Should see Streamlit app

---

## STEP 5: Setup CI/CD with Cloud Build Triggers

**In Google Cloud Console:**

1. Go to: **Cloud Build → Triggers**
2. Click: **"Create Trigger"**
3. Configure:
   - **Name:** deploy-doc-understand
   - **Event:** Push to branch
   - **Source:** Connect your GitHub repository
   - **Branch:** ^main$ (or ^master$)
   - **Configuration:** Cloud Build configuration file (cloudbuild.yaml)
   - **Location:** Repository (cloudbuild.yaml in root)

4. **Advanced (Substitution variables - optional):**
   - _SERVICE_NAME: doc-understand
   - _REGION: us-central1

5. Click: **"Create"**

**Result:** Every push to main branch automatically deploys!

---

## STEP 6: Connect GitHub Repository

**First-time setup:**

1. In Cloud Build → Triggers → Create Trigger
2. Click: **"Connect Repository"**
3. Select: **GitHub**
4. Authorize: Google Cloud Build to access your GitHub
5. Select: Your repository (doc-understand)
6. Connect

**After this:** Cloud Build can access your repo and auto-deploy on push

---

## STEP 7: Configure Cloud Run Service Settings

**Set environment variables:**
```bash
gcloud run services update doc-understand \
  --set-env-vars USE_GCS_OUTPUT=true,WRITE_LOCAL_COPY=false \
  --region us-central1
```

**Add secrets (if using Secret Manager):**
```bash
gcloud run services update doc-understand \
  --update-secrets GROQ_API_KEY=groq-api-key:latest \
  --update-secrets AWS_ACCESS_KEY_ID=aws-access-key:latest \
  --region us-central1
```

---

## STEP 8: Verify Deployment Works

**Test the deployed app:**

1. **Get URL:**
```bash
gcloud run services describe doc-understand \
  --region us-central1 \
  --format 'value(status.url)'
```

2. **Open in browser**

3. **Test functionality:**
   - Upload PDF
   - Ask question
   - Verify answer

4. **Check logs:**
```bash
gcloud run services logs read doc-understand --region us-central1
```

---

## STEP 9: Test CI/CD Automation

**Make a test change:**

1. Edit any file (e.g., add comment in streamlit_app.py)
2. Commit and push:
```bash
git add .
git commit -m "Test automated deployment"
git push origin main
```

3. **Watch Cloud Build:**
   - Go to: Cloud Build → History
   - Should see build starting automatically
   - Monitor progress

4. **Verify deployment:**
   - Wait for build to complete (~5-10 minutes)
   - Refresh Cloud Run URL
   - Should see updated version

---

# ENVIRONMENT VARIABLES REFERENCE

**Required in Cloud Run:**

| Variable | Purpose | Example |
|----------|---------|---------|
| GROQ_API_KEY | LLM API access | gsk_xxx... |
| AWS_ACCESS_KEY_ID | Textract access | AKIA... |
| AWS_SECRET_ACCESS_KEY | Textract access | xxx... |
| GCP_PROJECT_ID | Monitoring | doc-understand |
| USE_GCS_OUTPUT | Storage mode | true |

**Optional:**

| Variable | Purpose | Default |
|----------|---------|---------|
| PORT | Server port | 8080 |
| WRITE_LOCAL_COPY | Dual storage | false |

---

# TROUBLESHOOTING

## Build Fails

**Error:** "requirements.txt not found"
- **Fix:** Ensure `aws_extraction_requirements.txt` in root
- **Fix:** Dockerfile references correct filename

**Error:** "Permission denied"
- **Fix:** Check GCP service account has necessary roles
- **Fix:** Enable all required APIs

## Deployment Fails

**Error:** "Service failed to start"
- **Check:** Logs in Cloud Run console
- **Check:** Port 8080 configured correctly
- **Check:** Streamlit command path correct

**Error:** "Out of memory"
- **Fix:** Increase memory in cloudbuild.yaml (currently 2Gi)
- **Fix:** Or in deploy command: --memory 4Gi

## App Crashes After Deployment

**Error:** "Module not found"
- **Fix:** Add missing package to aws_extraction_requirements.txt
- **Fix:** Rebuild and redeploy

**Error:** "Authentication failed"
- **Fix:** Check environment variables set correctly
- **Fix:** Verify secrets in Secret Manager

## Secrets Not Working

**Check environment variables:**
```bash
gcloud run services describe doc-understand \
  --region us-central1 \
  --format 'value(spec.template.spec.containers[0].env)'
```

---

# MONITORING DEPLOYED APP

**Access Cloud Monitoring:**
```
https://console.cloud.google.com/monitoring/dashboards?project=doc-understand
```

**Metrics to Watch:**
- Query response time (should be <10 seconds)
- Success rate (should be >95%)
- RAG quality score (should be >0.75)
- Error count (should be minimal)

**Alerts:**
- Configured to email: gujjula.yaswanth@northeastern.edu
- On: Slow queries, high errors, poor quality

---

# COST ESTIMATE

**Cloud Run Pricing (as of Dec 2025):**
- Free tier: First 2 million requests/month
- After free tier: ~$0.00002 per request
- Memory: ~$0.0000025 per GB-second
- CPU: ~$0.00001 per vCPU-second

**Expected Monthly Cost:**
- Low usage (100 queries/day): **$0-5**
- Medium usage (1000 queries/day): **$10-20**
- High usage (10000 queries/day): **$50-100**

**Your monitoring is FREE** (within GCP free tier limits)

---

# REPLICATION INSTRUCTIONS

**For someone to replicate your deployment:**

1. Clone repository
2. Copy .env.example to .env and fill in secrets
3. Run: `gcloud run deploy doc-understand --source .`
4. Access URL provided
5. Done!

**Time:** 10-15 minutes for fresh deployment

---

# UPDATING THE DEPLOYMENT

**Manual update:**
```bash
gcloud run deploy doc-understand --source . --region us-central1
```

**Automatic update:**
- Just push code to GitHub main branch
- Cloud Build trigger automatically deploys
- No manual intervention needed

---

# ROLLBACK

**If deployment breaks:**

```bash
# List revisions
gcloud run revisions list --service doc-understand --region us-central1

# Rollback to previous revision
gcloud run services update-traffic doc-understand \
  --to-revisions REVISION_NAME=100 \
  --region us-central1
```

---

**END OF DEPLOYMENT GUIDE**
