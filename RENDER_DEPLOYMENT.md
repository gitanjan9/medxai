# 🚀 Render Deployment Guide

## Prerequisites

1. **GitHub/GitLab account** with your repository pushed
2. **Render account** (free tier available at render.com)
3. **Model artifacts** ready in `/artifacts` directory

## 🔒 Security Checklist (CRITICAL)

### ✅ Before Deploying:

- [ ] **NEVER commit `.env` file** (already in `.gitignore`)
- [ ] **Remove all hardcoded secrets** from code
- [ ] **Generate new JWT_SECRET_KEY**: `openssl rand -hex 32`
- [ ] **Use strong passwords** for admin and database
- [ ] **Review CORS_ORIGINS** to only allow your frontend domain
- [ ] **Disable public registration** in production (`MEDXAI_OPEN_REGISTRATION=false`)

### 🚨 Vulnerabilities Fixed:

1. ✅ Removed hardcoded HuggingFace token from `.env`
2. ✅ Replaced weak passwords (`admin123`, `medxai123`)
3. ✅ Updated `.env` to use placeholder values
4. ✅ Added CORS configuration to Render config
5. ✅ Set `COOKIE_SECURE=true` for production
6. ✅ Disabled public registration by default

---

## 📋 Deployment Steps

### 1. Prepare Your Repository

```bash
# Ensure .env is NOT committed
git status | grep .env  # Should show nothing

# Commit your changes
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create Render Services

#### Option A: Blueprint (Automated - Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New" → "Blueprint"**
3. Connect your GitHub/GitLab repository
4. Render will detect `render.yaml` and create all services automatically
5. **Set required environment variables** (see below)

#### Option B: Manual Setup

1. **Create PostgreSQL Database:**
   - Name: `medicalxai-postgres`
   - Database: `medxai`
   - User: `medxai`
   - Plan: Starter (upgrade to Standard for production)

2. **Create Redis Instance:**
   - Name: `medicalxai-redis`
   - Plan: Free or Starter

3. **Create Web Service (Backend):**
   - Name: `medicalxai-api`
   - Runtime: Docker
   - Dockerfile path: `./deployment/Dockerfile.api`
   - Plan: Standard (2 vCPU, 4 GB RAM - required for PyTorch)
   - Add persistent disk: 10 GB at `/app/artifacts`

4. **Create Static Site (Frontend):**
   - Name: `medicalxai-frontend`
   - Build command: `cd frontend && npm ci && npm run build`
   - Publish directory: `./frontend/dist`

### 3. Configure Environment Variables

In Render Dashboard → `medicalxai-api` → Environment:

#### 🔐 Required Secrets (Set Manually):

```bash
# Generate with: openssl rand -hex 32
JWT_SECRET_KEY=<your-generated-secret-here>

# Admin credentials (change immediately after first login)
MEDXAI_ADMIN_EMAIL=admin@yourdomain.com
MEDXAI_ADMIN_PASSWORD=<strong-password-here>

# HuggingFace token (if needed for model downloads)
HF_TOKEN=<your-hf-token>
```

#### ✅ Auto-configured (from render.yaml):

- `DATABASE_URL` - Auto-linked from PostgreSQL
- `REDIS_URL` - Auto-linked from Redis
- `MEDXAI_ADMIN_SECRET` - Auto-generated
- Model configuration variables
- CORS settings

#### 🌐 Update CORS Origins:

```bash
# Replace with your actual frontend URL
CORS_ORIGINS=https://medicalxai-frontend.onrender.com
```

### 4. Upload Model Artifacts

Since Render has a persistent disk mounted at `/app/artifacts`, you need to upload your model files:

**Option 1: Use Render Shell**
```bash
# In Render Dashboard → medicalxai-api → Shell
cd /app/artifacts
# Upload files via SCP or download from cloud storage
```

**Option 2: Download on First Run**
Add a startup script in your Dockerfile to download artifacts from cloud storage (S3, GCS, etc.)

### 5. Deploy

```bash
# Render will auto-deploy on git push
git push origin main

# Or manually trigger deploy in Render Dashboard
```

### 6. Verify Deployment

```bash
# Check health endpoint
curl https://medicalxai-api.onrender.com/health

# Check API docs
open https://medicalxai-api.onrender.com/docs
```

---

## 🔧 Configuration Reference

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | JWT signing key (64 chars hex) | `openssl rand -hex 32` |
| `MEDXAI_ADMIN_EMAIL` | Initial admin email | `admin@yourdomain.com` |
| `MEDXAI_ADMIN_PASSWORD` | Initial admin password | Strong password |
| `DATABASE_URL` | PostgreSQL connection string | Auto-set by Render |
| `CORS_ORIGINS` | Allowed frontend origins | `https://yourfrontend.com` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDXAI_OPEN_REGISTRATION` | `false` | Allow public user registration |
| `WEB_CONCURRENCY` | `1` | Number of Gunicorn workers |
| `COOKIE_SECURE` | `true` | Require HTTPS for cookies |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | JWT refresh token lifetime |
| `HF_TOKEN` | - | HuggingFace API token |

---

## 📊 Resource Requirements

### Minimum (Render Standard Plan):
- **CPU**: 2 vCPU
- **RAM**: 4 GB
- **Disk**: 10 GB (for model artifacts)
- **Database**: Starter plan (1 GB RAM)

### Recommended (Production):
- **CPU**: 4 vCPU
- **RAM**: 8 GB
- **Disk**: 20 GB
- **Database**: Standard plan (4 GB RAM)

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" during model loading

**Solution:**
- Upgrade to a larger plan (8 GB RAM)
- Or reduce `WEB_CONCURRENCY` to `1`
- Or use a smaller model architecture

### Issue: "Database connection failed"

**Solution:**
- Verify `DATABASE_URL` is set correctly
- Check PostgreSQL service is running
- Ensure database is in the same region as API

### Issue: "CORS error" from frontend

**Solution:**
- Update `CORS_ORIGINS` to include your frontend URL
- Ensure URL includes protocol (`https://`)
- No trailing slash in URL

### Issue: "Model artifacts not found"

**Solution:**
- Upload artifacts to persistent disk
- Or configure auto-download from cloud storage
- Check disk is mounted at `/app/artifacts`

---

## 🔐 Security Best Practices

1. **Rotate secrets regularly**
   - JWT keys every 90 days
   - Database passwords every 180 days

2. **Enable rate limiting**
   - Already configured via `slowapi`
   - Monitor in Render logs

3. **Use HTTPS only**
   - `COOKIE_SECURE=true` (enforced in render.yaml)
   - Render provides free SSL certificates

4. **Restrict registration**
   - `MEDXAI_OPEN_REGISTRATION=false` for production
   - Use admin panel to invite users

5. **Monitor logs**
   - Check Render Dashboard → Logs regularly
   - Set up alerts for errors

6. **Database backups**
   - Render PostgreSQL includes daily backups
   - Upgrade to Standard plan for point-in-time recovery

---

## 💰 Cost Estimate

### Free Tier:
- Static site (frontend): **$0/month**
- PostgreSQL Starter: **$7/month**
- Redis Free: **$0/month**
- Web Service Standard: **$25/month**

**Total: ~$32/month**

### Production Tier:
- Static site: **$0/month**
- PostgreSQL Standard: **$20/month**
- Redis Starter: **$10/month**
- Web Service (4 vCPU, 8 GB): **$85/month**

**Total: ~$115/month**

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [PostgreSQL on Render](https://render.com/docs/databases)
- [Docker on Render](https://render.com/docs/docker)

---

## 🆘 Support

If you encounter issues:
1. Check Render logs: Dashboard → Service → Logs
2. Review this guide's troubleshooting section
3. Check Render status: https://status.render.com
4. Contact Render support: https://render.com/support
