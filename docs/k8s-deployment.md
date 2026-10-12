# ☸️ Local Kubernetes Deployment Guide

This guide provides end-to-end instructions for deploying the **PolicyGPT** application (FastAPI + LangGraph + PGVector + Groq 120B) to a local Kubernetes cluster.

---

## 📑 Prerequisites

Before deploying, make sure you have:
1. **Docker Desktop** installed on Windows/Mac/Linux.
2. **Kubernetes Enabled** in Docker Desktop:
   - Open **Docker Desktop Settings** ➔ Select **Kubernetes** ➔ Check **"Enable Kubernetes"** ➔ Click **Apply & restart**.
   *(Alternatively, you can use **Minikube** or **Kind**).*
3. **`kubectl`** CLI configured:
   ```bash
   kubectl cluster-info
   ```
4. A free **Groq API Key** from [console.groq.com/keys](https://console.groq.com/keys).

---

## 🏗️ Kubernetes Manifest Structure

The manifests in the `k8s/` directory:

```
k8s/
├── namespace.yaml             # Defines 'rag-app' namespace
├── 01-configmap-secrets.yaml  # ConfigMap (DB URLs, models), Secrets (API keys), and init-db SQL
├── 02-postgres.yaml           # PVC (5Gi storage), pgvector/pgvector:pg16 Deployment & Service
├── 03-app.yaml                # PolicyGPT FastAPI Deployment, Probes & NodePort Service (30080)
└── kustomization.yaml         # Kustomize manifest for single-command deploy
```

---

## 🚀 Step-by-Step Deployment Instructions

### Step 1: Build the Application Docker Image Locally

Build the container image so it is available to your local Kubernetes engine:

```bash
# In the project root directory:
docker build -t airline-rag-app:latest .
```

> **Minikube Users Note:** If using Minikube with the Docker driver, load the image into the Minikube cache:
> ```bash
> minikube image load airline-rag-app:latest
> ```

---

### Step 2: Configure Your Secrets

Open [k8s/01-configmap-secrets.yaml](file:///d:/rag-langgraph-k8s-quickstart/k8s/01-configmap-secrets.yaml) and replace the placeholder Groq API key with your actual key:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-app-secrets
  namespace: rag-app
type: Opaque
stringData:
  GROQ_API_KEY: "gsk_your_actual_groq_api_key_here"  # <-- Paste your key here
  ADMIN_API_KEY: "admin-secret-key-123"
  POSTGRES_USER: "postgres"
  POSTGRES_PASSWORD: "postgrespassword"
```

*(Alternatively, create the secret directly using kubectl):*
```bash
kubectl create namespace rag-app --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic rag-app-secrets \
  --from-literal=GROQ_API_KEY="your_groq_api_key" \
  --from-literal=ADMIN_API_KEY="admin-secret-key-123" \
  --from-literal=POSTGRES_USER="postgres" \
  --from-literal=POSTGRES_PASSWORD="postgrespassword" \
  -n rag-app
```

---

### Step 3: Deploy All Resources to Kubernetes

You can deploy everything using **Kustomize** with a single command:

```bash
kubectl apply -k k8s/
```

Or apply the manifests individually in order:

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. Config & Secrets
kubectl apply -f k8s/01-configmap-secrets.yaml

# 3. PostgreSQL with PGVector & Storage
kubectl apply -f k8s/02-postgres.yaml

# 4. PolicyGPT Application
kubectl apply -f k8s/03-app.yaml
```

---

### Step 4: Verify Pod Status

Check that the PostgreSQL and Application Pods are running:

```bash
kubectl get pods -n rag-app -w
```

Expected output:
```
NAME                               READY   STATUS    RESTARTS   AGE
postgres-5c4dbd846-x8r7f           1/1     Running   0          45s
airline-rag-app-796fbdbbf5-t678b   1/1     Running   0          30s
```

Verify services:
```bash
kubectl get svc -n rag-app
```

Expected output:
```
NAME                      TYPE        CLUSTER-IP       PORT(S)          AGE
postgres-service          ClusterIP   10.96.14.21      5432/TCP         1m
airline-rag-app-service   NodePort    10.102.215.12    8000:30080/TCP   1m
```

---

### Step 5: Access the Web UI

You have two convenient ways to access the application:

#### Method A: Port-Forwarding (Recommended)
Forward port 8000 from the cluster service to your local machine:

```bash
kubectl port-forward svc/airline-rag-app-service 8000:8000 -n rag-app
```

Now open your browser: **`http://localhost:8000/`** 🎉

#### Method B: Direct NodePort
Because `airline-rag-app-service` is exposed as a `NodePort` on port `30080`:
- **Docker Desktop**: Open **`http://localhost:30080/`**
- **Minikube**: Run `minikube service airline-rag-app-service -n rag-app`

---

### Step 6: Initialize and Index Airline Documents

Once the application is up, initialize the database schema and index documents:

```bash
# Initialize database extensions & schemas
curl -X POST -H "x-api-key: admin-secret-key-123" http://localhost:8000/api/admin/initialize_databases

# Index all documents in documents/ directory
curl -X POST -H "x-api-key: admin-secret-key-123" http://localhost:8000/api/admin/index_all_documents
```

---

## 🔍 Useful Diagnostic & Debugging Commands

```bash
# View application logs in real-time
kubectl logs -f deployment/airline-rag-app -n rag-app

# View PostgreSQL logs
kubectl logs -f deployment/postgres -n rag-app

# Check detailed Pod status / events if pending or failing
kubectl describe pod -l app=airline-rag-app -n rag-app

# Access bash inside the app container
kubectl exec -it deployment/airline-rag-app -n rag-app -- /bin/bash

# Access PostgreSQL via psql inside the cluster
kubectl exec -it deployment/postgres -n rag-app -- psql -U postgres -d vectordb
```

---

## 🧹 Cleanup / Teardown

To delete all deployed resources and free up storage:

```bash
# Delete all resources via Kustomize
kubectl delete -k k8s/

# Or delete the entire namespace
kubectl delete namespace rag-app
```
