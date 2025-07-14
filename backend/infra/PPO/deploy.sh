#!/bin/bash

# Exit immediately on error
set -e

# Azure ML configuration
resource_group="qunta-ai-group"
workspace_name="qunta-ml-workspace"
compute_name="gpu-spot-cluster"
vm_size="Standard_NC6s_v3"
location="eastus"  # Change if needed

echo "🔧 Logging into Azure..."
az login

echo "📁 Setting default resource group and workspace..."
az configure --defaults group=$resource_group workspace=$workspace_name

echo "🚀 Creating Azure ML workspace (if it doesn't exist)..."
az ml workspace create --name $workspace_name --resource-group $resource_group --location $location || true

echo "🧠 Creating GPU spot compute cluster: $compute_name"
az ml compute create \
  --name $compute_name \
  --type AmlCompute \
  --min-instances 0 \
  --max-instances 1 \
  --size $vm_size \
  --tier low_priority \ 
  --resource-group $resource_group \
  --workspace-name $workspace_name

echo "✅ Spot compute cluster '$compute_name' is ready to use!"
