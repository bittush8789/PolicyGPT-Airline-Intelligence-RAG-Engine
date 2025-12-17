# variables.tf - Define all variables with descriptions and defaults

variable "linode_token" {
  description = "Linode API Token with appropriate permissions"
  type        = string
  sensitive   = true
  # Set via environment variable: export TF_VAR_linode_token="your-token"
  # Or pass via command line: terraform apply -var="linode_token=your-token"
}

variable "region" {
  description = "Akamai Cloud region for all resources. Must support Linodes, Managed Databases, Kubernetes, and Object Storage."
  type        = string
  default     = "us-sea"
  
  validation {
    condition = contains([
      "ap-south",    # Singapore, SG
      "au-mel",      # Melbourne, AU
      "br-gru",      # Sao Paulo, BR
      "es-mad",      # Madrid, ES
      "eu-central",  # Frankfurt, DE
      "fr-par",      # Paris, FR
      "gb-lon",      # London 2, UK
      "id-cgk",      # Jakarta, ID
      "in-maa",      # Chennai, IN
      "it-mil",      # Milan, IT
      "jp-osa",      # Osaka, JP
      "jp-tyo-3",    # Tokyo 3, JP
      "nl-ams",      # Amsterdam, NL
      "se-sto",      # Stockholm, SE
      "sg-sin-2",    # Singapore 2, SG
      "us-east",     # Newark, NJ
      "us-iad",      # Washington, DC
      "us-lax",      # Los Angeles, CA
      "us-mia",      # Miami, FL
      "us-ord",      # Chicago, IL
      "us-sea",      # Seattle, WA
      "us-southeast" # Atlanta, GA
    ], var.region)
    error_message = "The selected region '${var.region}' must support all required capabilities: Linodes, Managed Databases, Kubernetes, and Object Storage. Valid regions: ap-south, au-mel, br-gru, es-mad, eu-central, fr-par, gb-lon, id-cgk, in-maa, it-mil, jp-osa, jp-tyo-3, nl-ams, se-sto, sg-sin-2, us-east, us-iad, us-lax, us-mia, us-ord, us-sea, us-southeast."
  }
}

variable "cluster_label" {
  description = "Label for the LKE cluster"
  type        = string
  default     = "rag-chatbot-cluster"
  
  validation {
    condition     = length(var.cluster_label) > 0 && length(var.cluster_label) <= 32
    error_message = "Cluster label must be between 1 and 32 characters."
  }
}

variable "postgres_vector_label" {
  description = "Label for the vector database"
  type        = string
  default     = "rag-vector-db"
}

variable "postgres_conversation_label" {
  description = "Label for the conversation database"
  type        = string
  default     = "rag-conversation-db"
}

variable "bucket_label" {
  description = "Label for the object storage bucket"
  type        = string
  default     = "rag-documents"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.bucket_label))
    error_message = "Bucket label must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "postgres_type" {
  description = "Linode instance type for PostgreSQL databases"
  type        = string
  default     = "g6-standard-2"  # 4GB Shared
  
  # Common options:
  # g6-nanode-1     - 1GB Shared
  # g6-standard-1   - 2GB Shared
  # g6-standard-2   - 4GB Shared
  # g6-dedicated-2  - 4GB Dedicated
  # g6-dedicated-4  - 8GB Dedicated
}

variable "lke_node_type" {
  description = "Linode instance type for LKE nodes"
  type        = string
  default     = "g6-dedicated-4"  # 8GB Dedicated
}

variable "lke_node_count" {
  description = "Initial number of nodes in the LKE cluster"
  type        = number
  default     = 3
  
  validation {
    condition     = var.lke_node_count >= 1 && var.lke_node_count <= 100
    error_message = "Node count must be between 1 and 100."
  }
}

variable "lke_autoscaler_min" {
  description = "Minimum number of nodes for autoscaling"
  type        = number
  default     = 3
}

variable "lke_autoscaler_max" {
  description = "Maximum number of nodes for autoscaling"
  type        = number
  default     = 5
}

variable "k8s_version" {
  description = "Kubernetes version for LKE cluster"
  type        = string
  default     = "1.34"
  # Check available versions: linode-cli lke versions-list
}

variable "enable_ha_control_plane" {
  description = "Enable High Availability for Kubernetes control plane"
  type        = bool
  default     = true
}

variable "enable_bucket_versioning" {
  description = "Enable versioning for the object storage bucket"
  type        = bool
  default     = true
}

variable "postgres_allow_list" {
  description = "List of IP addresses/CIDR blocks allowed to connect to PostgreSQL"
  type        = list(string)
  default     = []
  
  # For production, specify your IPs:
  # default = ["1.2.3.4/32", "5.6.7.0/24"]
  # Note that LKE node IPs will automatically be added to allow list
}

variable "backup_day" {
  description = "Day of week for automated backups (sunday, monday, etc.)"
  type        = string
  default     = "sunday"
}

variable "backup_hour" {
  description = "Hour of day (0-23) for automated backups"
  type        = number
  default     = 2
  
  validation {
    condition     = var.backup_hour >= 0 && var.backup_hour <= 23
    error_message = "Backup hour must be between 0 and 23."
  }
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = list(string)
  default     = ["rag-chatbot", "production", "terraform"]
}

variable "ssl_connection" {
  description = "Require SSL connections to PostgreSQL databases"
  type        = bool
  default     = true
}
