# Automatisiertes Setup für OpenManus-RL

Dieses Skript automatisiert die Einrichtung des Projekts nach Originalstruktur inkl. Supabase, MCP-Server und Open-Source-Frameworks.

## Schritte
1. Supabase CLI installieren
2. Supabase-Projekt verlinken und Migrationen ausführen
3. Edge Functions deployen
4. Python-Abhängigkeiten installieren
5. Webshop-Umgebung einrichten
6. MCP-Server starten (optional)

## Bash-Skript
```bash
# Supabase CLI installieren
npm install -g supabase

# Projekt verlinken
supabase link --project-ref jdjhkmenfkmbaeaskkug

# Migrationen ausführen
supabase db push

# Edge Functions deployen
supabase functions deploy submit-rollout
supabase functions deploy log-agent-state
supabase functions deploy complete-rollout
supabase functions deploy get-metrics

# Python-Abhängigkeiten installieren
pip install -r requirements.txt
pip install supabase

# Webshop-Umgebung einrichten
cd openmanus_rl/environments/env_package/webshop/webshop/
conda create -n agentenv_webshop python=3.10 -y
conda activate agentenv_webshop
bash ./setup.sh -d all

# MCP-Server (optional, falls vorhanden)
# python mcp_server.py
```

## Hinweise
- Das Skript kann als `setup_all.sh` im Projekt abgelegt werden.
- Für Windows kann eine entsprechende PowerShell-Version bereitgestellt werden.
- Weitere Umgebungen und Frameworks können nach Bedarf ergänzt werden.
