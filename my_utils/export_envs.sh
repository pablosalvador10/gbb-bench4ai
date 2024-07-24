#!/bin/bash
# export_envs.sh
grep -v '^#' .env | sed 's/^/export /' >> env.mk
echo "Generated env.mk with exported variables"
