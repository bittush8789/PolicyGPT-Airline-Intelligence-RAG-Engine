# Terraform configuration for RAG Chatbot infrastructure on Linode
# This creates all required resources in us-lax region

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    linode = {
      source  = "linode/linode"
      version = "~> 2.0"
    }
  }
}

provider "linode" {
  token = var.linode_token
}

# PostgreSQL Database for Vector Storage
# Note: Vector extension creation is handled by the Python admin API
# See app/api/admin.py for the initialize_vector_database method
resource "linode_database_postgresql_v2" "vector_db" {
  label      = var.postgres_vector_label
  region     = var.region
  engine_id  = "postgresql/18"
  type       = var.postgres_type
  allow_list = var.postgres_allow_list
}

# PostgreSQL Database for Conversation Storage
# Note: Schema initialization (tables, indexes, functions) is handled by the Python admin API
# See app/api/admin.py for the initialize_state_database method
resource "linode_database_postgresql_v2" "conversation_db" {
  label      = var.postgres_conversation_label
  region     = var.region
  engine_id  = "postgresql/18"
  type       = var.postgres_type
  allow_list = var.postgres_allow_list
}

# Object Storage Bucket for Documents
resource "linode_object_storage_bucket" "documents" {
  label  = var.bucket_label
  region = var.region
}

# Object Storage Key for accessing the bucket
resource "linode_object_storage_key" "bucket_key" {
  label = "${var.bucket_label}-access-key"
  
  bucket_access {
    bucket_name = var.bucket_label
    region      = var.region
    permissions = "read_write"
  }
  
  depends_on = [
    linode_object_storage_bucket.documents
  ]
}

#### LKE Cluster
resource "linode_lke_cluster" "rag_cluster" {
  label       = var.cluster_label
  k8s_version = var.k8s_version
  region      = var.region
  
  # Enable App Platform Layer (APL)
  apl_enabled = true
  
  # High Availability control plane
  control_plane {
    high_availability = var.enable_ha_control_plane
  }
  
  tags = var.tags
  
  pool {
    type  = var.lke_node_type
    count = var.lke_node_count
    
    autoscaler {
      min = var.lke_autoscaler_min
      max = var.lke_autoscaler_max
    }
  }
}

# Kubernetes provider configuration (for post-setup if needed)
# Note: You'll need to export KUBECONFIG or configure kubectl manually
resource "local_file" "kubeconfig" {
  content  = base64decode(linode_lke_cluster.rag_cluster.kubeconfig)
  filename = "${path.module}/kubeconfig.yaml"
  
  file_permission = "0600"
}

# Outputs
output "VECTOR_DB_CONNECTION_STRING" {
  description = "Vector database connection string"
  value       = "postgresql://${linode_database_postgresql_v2.vector_db.root_username}:${linode_database_postgresql_v2.vector_db.root_password}@${linode_database_postgresql_v2.vector_db.host_primary}:${linode_database_postgresql_v2.vector_db.port}/defaultdb?sslmode=require"
  sensitive   = true
}

output "CONVERSATION_DB_CONNECTION_STRING" {
  description = "Conversation database connection string"
  value       = "postgresql://${linode_database_postgresql_v2.conversation_db.root_username}:${linode_database_postgresql_v2.conversation_db.root_password}@${linode_database_postgresql_v2.conversation_db.host_primary}:${linode_database_postgresql_v2.conversation_db.port}/defaultdb?sslmode=require"
  sensitive   = true
}

output "S3_ENDPOINT" {
  description = "S3-compatible endpoint for object storage"
  value       = linode_object_storage_bucket.documents.s3_endpoint
}

output "S3_BUCKET_NAME" {
  description = "Object storage bucket name"
  value       = linode_object_storage_bucket.documents.label
}

output "S3_ACCESS_KEY" {
  description = "Object storage access key"
  value       = linode_object_storage_key.bucket_key.access_key
  sensitive   = true
}

output "S3_SECRET_KEY" {
  description = "Object storage secret key"
  value       = linode_object_storage_key.bucket_key.secret_key
  sensitive   = true
}

output "LKE_CLUSTER_ID" {
  description = "LKE Cluster ID"
  value       = linode_lke_cluster.rag_cluster.id
}

output "LKE_CLUSTER_KUBECONFIG" {
  description = "Kubernetes config for the LKE cluster"
  value       = linode_lke_cluster.rag_cluster.kubeconfig
  sensitive   = true
}

output "KUBECONFIG_PATH" {
  description = "Path to the generated kubeconfig file"
  value       = local_file.kubeconfig.filename
}
