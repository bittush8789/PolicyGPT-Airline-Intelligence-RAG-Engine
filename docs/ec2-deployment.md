# ☁️ AWS EC2 Production Deployment Guide

This guide provides a comprehensive, step-by-step walkthrough for deploying the **PolicyGPT Airline RAG Chatbot** stack onto an **AWS EC2 (Ubuntu Linux)** virtual machine using Docker, Docker Compose, Nginx reverse proxy, and Let's Encrypt SSL.

---

## 🏗️ Architecture on AWS EC2

```
                                  AWS Cloud
┌───────────────────────────────────────────────────────────────────────────────┐
│  EC2 Security Group (Ports: 22, 80, 443)                                      │
│                                                                               │
│  [ Users / Web Browsers ]                                                     │
│             │                                                                 │
│             │  HTTPS (Port 443) / Let's Encrypt SSL                           │
│             ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    Nginx Reverse Proxy (Host Port 80 / 443)             │  │
│  │       • SSL Termination                                                 │  │
│  │       • SSE Streaming Support (`proxy_buffering off`)                   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                         │
│                      HTTP Proxy     │ (http://localhost:8000)                 │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  Docker Compose Network (`rag-network`)                                 │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────┐         ┌───────────────────────────────┐  │  │
│  │  │     airline-rag-app     │         │     airline-rag-postgres      │  │  │
│  │  │        (FastAPI)        │◄───────►│      (pgvector/pgvector:16)   │  │  │
│  │  │                         │         │                               │  │  │
│  │  │  • Guardrails & Evals   │         │  • vectordb (HNSW index)      │  │  │
│  │  │  • FastEmbed (ONNX)     │         │  • statedb (LangGraph memory) │  │  │
│  │  └───────────┬─────────────┘         └───────────────┬───────────────┘  │  │
│  └──────────────┼───────────────────────────────────────┼──────────────────┘  │
│                 │                                       │                     │
└─────────────────┼───────────────────────────────────────┼─────────────────────┘
                  │ Outbound HTTPS                        │ Mount
                  ▼                                       ▼
       ┌────────────────────────┐              ┌──────────────────────┐
       │   Groq Cloud LPU API   │              │   EBS Volume (gp3)   │
       │ (openai/gpt-oss-120b)  │              │ (Persistent DB Data) │
       │           &            │              └──────────────────────┘
       │  LangSmith Monitoring  │
       └────────────────────────┘
```

---

## 📋 Recommended EC2 Specifications

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Instance Type** | `t3.medium` (2 vCPU, 4 GiB RAM) | **`t3.large`** (2 vCPU, 8 GiB RAM) or `c6i.large` |
| **Operating System** | Ubuntu 22.04 LTS / 24.04 LTS (x86_64) | Ubuntu 24.04 LTS (x86_64) |
| **Storage (EBS)** | 20 GB gp3 SSD | 30–50 GB gp3 SSD (3000 IOPS) |
| **Public IP** | Standard Auto-assigned Public IPv4 | **AWS Elastic IP** (retains IP across restarts) |

> 💡 **Why `t3.large`?**
> Local FastEmbed embeddings (`BAAI/bge-small-en-v1.5` running via ONNX Runtime) and in-memory batch document chunking run fastest and never risk Linux OOM-killer (Out-Of-Memory) pauses when provided with 4–8 GB RAM.

---

## 🔒 Step 1: AWS Security Group Configuration

In the AWS EC2 Management Console, create or edit your Security Group with these **Inbound Rules**:

| Type | Protocol | Port Range | Source | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | `22` | `My IP` (or VPN CIDR) | Secure SSH terminal access |
| **HTTP** | TCP | `80` | `0.0.0.0/0` | Let's Encrypt verification & HTTP to HTTPS redirect |
| **HTTPS** | TCP | `443` | `0.0.0.0/0` | Encrypted public access to PolicyGPT UI & APIs |
| *(Optional dev)* | TCP | `8000` | `My IP` | Direct FastAPI debugging (disable in production) |

> ⚠️ **CRITICAL SECURITY RULE:** Never expose Port `5432` (PostgreSQL) to `0.0.0.0/0`. PostgreSQL is accessed exclusively within the internal Docker network.

---

## 💻 Step 2: Connect to EC2 & Configure System

Connect to your EC2 instance via SSH:

```bash
ssh -i /path/to/your-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

### 1. Update Packages & Setup Swap Space (Recommended)
Creating a 4 GB swap file prevents out-of-memory crashes during model caching or heavy multi-user RAG queries:

```bash
sudo apt update && sudo apt upgrade -y

# Allocate 4GB Swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 🐳 Step 3: Install Docker & Docker Compose

Run the official Docker installation commands on Ubuntu:

```bash
# 1. Install prerequisites
sudo apt install -y ca-certificates curl gnupg lsb-release git

# 2. Add Docker official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 3. Set up repository
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Install Docker Engine and Docker Compose Plugin
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# 6. Allow 'ubuntu' user to run docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

---

## 📦 Step 4: Clone the Project & Configure Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/rag-langgraph-k8s-quickstart.git /home/ubuntu/rag-app
cd /home/ubuntu/rag-app

# Create .env from template
cp .env.example .env
```

### Configure Production Secrets in `.env`:

Edit the file with `nano .env`:

```bash
nano .env
```

Set the production variables:

```dotenv
# ==============================================================================
# AWS EC2 Production Environment Configuration
# ==============================================================================

# Admin API Key (Choose a long, random secret key for managing index & DBs)
ADMIN_API_KEY=prod-secret-admin-key-change-this-98765

# Groq Cloud API Key (Get at https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Groq LLM & Local FastEmbed Model
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Database Settings (Uses internal Docker network name 'postgres')
VECTOR_DB_CONNECTION_STRING=postgresql://postgres:StrongProdPassword123!@postgres:5432/vectordb
CONVERSATION_DB_CONNECTION_STRING=postgresql://postgres:StrongProdPassword123!@postgres:5432/statedb

# Document Directory & Server Settings
DOCUMENTS_DIR=/app/documents
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# LangSmith Observability (Recommended for production monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_actual_langsmith_key_here
LANGCHAIN_PROJECT=policygpt-airline-rag-prod
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

> 🔐 **PostgreSQL Password Sync**: If you modify `StrongProdPassword123!`, also update `POSTGRES_PASSWORD` inside `docker-compose.yml`.

---

## 🚀 Step 5: Start Docker Stack

Build and start the application and PostgreSQL containers in background mode:

```bash
docker compose up -d --build
```

### Verify Container Status:
```bash
docker compose ps
```
Both `airline-rag-app` and `airline-rag-postgres` should show `Up` and `healthy`.

### Check Startup Logs:
```bash
docker compose logs -f app
```
You should observe:
```text
✓ GROQ_API_KEY: gsk_... (configured)
✓ LangSmith Tracing V2: ENABLED (Project: 'policygpt-airline-rag-prod')
✓ Vector store initialized successfully
✓ PostgreSQL checkpointer initialized successfully
Uvicorn running on http://0.0.0.0:8000
```
*(Press `Ctrl+C` to exit log viewing).*

---

## 🗄️ Step 6: Initialize Databases & Index Documents

Once containers are running, trigger database setup and document indexing:

```bash
# 1. Initialize databases and vector extension
curl -X POST \
  -H "x-api-key: prod-secret-admin-key-change-this-98765" \
  http://localhost:8000/api/admin/initialize_databases

# 2. Index all policy documents from /app/documents
curl -X POST \
  -H "x-api-key: prod-secret-admin-key-change-this-98765" \
  http://localhost:8000/api/admin/index_all_documents
```

Verify index count:
```bash
curl http://localhost:8000/api/airlines
```
Output will confirm **10 airlines** and **1,984 document chunks** indexed.

---

## 🌐 Step 7: Setup Nginx Reverse Proxy with SSE Streaming

Nginx acts as the front-facing web server, handles SSL, and proxies requests to FastAPI on port 8000.

### 1. Install Nginx:
```bash
sudo apt install -y nginx
```

### 2. Configure Nginx Site:
Create `/etc/nginx/sites-available/policygpt`:

```bash
sudo nano /etc/nginx/sites-available/policygpt
```

Paste the following configuration (replace `yourdomain.com` with your domain or EC2 Public DNS):

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com; # Or your EC2 Public DNS

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CRITICAL: Disable buffering for real-time Server-Sent Events (SSE) streaming
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

### 3. Enable Site & Test Nginx:
```bash
# Remove default site & link new site
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/policygpt /etc/nginx/sites-enabled/

# Test configuration syntax
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

---

## 🔒 Step 8: Configure Free HTTPS with Let's Encrypt (Certbot)

If you have a domain pointed to your EC2 Elastic IP:

```bash
# 1. Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# 2. Obtain and automatically apply SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Follow the prompts (enter email, accept terms). Certbot will automatically configure HTTPS on port 443 with automatic renewal!

Test certificate auto-renewal:
```bash
sudo certbot renew --dry-run
```

---

## 🛡️ Step 9: Production Systemd Autostart

Ensure the Docker stack automatically restarts if the EC2 instance reboots:

Create a systemd unit `/etc/systemd/system/rag-docker.service`:

```bash
sudo nano /etc/systemd/system/rag-docker.service
```

Add:
```ini
[Unit]
Description=PolicyGPT Airline RAG Docker Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/rag-app
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-docker.service
```

---

## 📊 Step 10: Monitoring & Maintenance

### 1. LangSmith Monitoring
Open [https://smith.langchain.com/](https://smith.langchain.com/) ➔ Project: **`policygpt-airline-rag-prod`**.
- View latency p50/p99 graphs.
- Inspect prompt tokens and Groq completion tokens.
- Review full retrieval similarity chunks and LLM guardrail passes.

### 2. Automated PostgreSQL Backups (Daily Cron Job)
Create a lightweight backup script:

```bash
sudo mkdir -p /var/backups/rag-db
sudo nano /usr/local/bin/backup-rag-db.sh
```

Paste:
```bash
#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/var/backups/rag-db"
docker exec airline-rag-postgres pg_dump -U postgres vectordb | gzip > "$BACKUP_DIR/vectordb_$TIMESTAMP.sql.gz"
docker exec airline-rag-postgres pg_dump -U postgres statedb | gzip > "$BACKUP_DIR/statedb_$TIMESTAMP.sql.gz"
# Keep last 7 days
find "$BACKUP_DIR" -type f -mtime +7 -name "*.sql.gz" -exec rm -f {} +
```

Make executable & add to crontab:
```bash
sudo chmod +x /usr/local/bin/backup-rag-db.sh
# Run every night at 2:00 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-rag-db.sh") | crontab -
```

---

## ❓ Troubleshooting FAQ

### 1. Streaming feels delayed or appears all at once
**Root Cause**: Nginx proxy buffering is enabled.  
**Fix**: Ensure `proxy_buffering off;` and `proxy_http_version 1.1;` are present in your Nginx `location /` configuration, then reload Nginx (`sudo nginx -s reload`).

### 2. Container exits with code 137 (OOM Killer)
**Root Cause**: The EC2 instance ran out of memory while loading FastEmbed embedding models.  
**Fix**:
1. Check available RAM: `free -m`.
2. Ensure you created the 4GB swap space in [Step 2](#1-update-packages--setup-swap-space-recommended).
3. If concurrent traffic is high, upgrade the EC2 instance from `t3.medium` to `t3.large`.

### 3. How to add new airline policy documents on EC2?
Place new `.pdf` or `.txt` files in `/home/ubuntu/rag-app/documents/` and run:
```bash
curl -X POST \
  -H "x-api-key: prod-secret-admin-key-change-this-98765" \
  http://localhost:8000/api/admin/index_all_documents
```

---

## 🎯 Verification Checklist

- [ ] EC2 Security Group allows ports 22, 80, 443.
- [ ] Docker & Docker Compose running.
- [ ] `.env` has valid `GROQ_API_KEY` and `LANGCHAIN_API_KEY`.
- [ ] Containers `airline-rag-app` and `airline-rag-postgres` are healthy.
- [ ] `/api/health?services=true` returns `"status": "healthy"`.
- [ ] 1,984 document chunks indexed.
- [ ] Nginx proxies port 80/443 to `localhost:8000` with `proxy_buffering off;`.
- [ ] SSL certificate active via Let's Encrypt.
