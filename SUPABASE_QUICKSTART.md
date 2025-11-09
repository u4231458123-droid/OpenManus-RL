# OpenManus-RL Supabase Quick Start

## 🚀 Schnellstart

Das OpenManus-RL-Projekt ist jetzt vollständig für Supabase konfiguriert!

### Voraussetzungen

- Node.js und npm installiert
- Python 3.10+ installiert
- Git konfiguriert
- Supabase-Account

### 1. Installation

```powershell
# Supabase CLI installieren
npm install -g supabase

# Python-Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. Konfiguration

Ihre `.env.supabase` ist bereits erstellt mit:
```
SUPABASE_URL=https://jdjhkmenfkmbaeaskkug.supabase.co
SUPABASE_ANON_KEY=sb_publishable_IJFhatPZZcKJfB8G5QC9Tg_TqP4nTcX
SUPABASE_SERVICE_ROLE_KEY=sbp_ed71b8e9dd2c7d7205d626b99ad63a218934e67c
```

**WICHTIG**: Setzen Sie Ihr Datenbankpasswort:
```
DATABASE_URL=postgresql://postgres:IHR_PASSWORD@db.jdjhkmenfkmbaeaskkug.supabase.co:5432/postgres
```

### 3. Deployment

#### Automatisches Deployment (empfohlen)

```powershell
.\scripts\deploy_supabase.ps1
```

#### Manuelles Deployment

```powershell
# Projekt verknüpfen
supabase link --project-ref jdjhkmenfkmbaeaskkug

# Migrationen anwenden
supabase db push

# Edge Functions deployen
supabase functions deploy submit-rollout
supabase functions deploy log-agent-state
supabase functions deploy complete-rollout
supabase functions deploy get-metrics

# Datasets hochladen
python scripts/upload_datasets.py
```

## 📊 Verwendung

### Python-Integration

```python
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager
from openmanus_rl.utils.supabase_storage import StorageManager

# Training Run erstellen
run = TrainingRunManager.create_run(
    name="alfworld-experiment",
    algorithm="gigpo",
    environment="alfworld",
    config={"learning_rate": 0.0001}
)

# Rollout erstellen und loggen
rollout = RolloutManager.create_rollout(
    training_run_id=run["id"],
    episode_number=1,
    environment="alfworld"
)

# Agent State loggen
state_id = RolloutManager.log_agent_state(
    rollout_id=rollout["id"],
    step_number=1,
    observation="You are in a room",
    action="go north"
)

# Checkpoint hochladen
StorageManager.upload_checkpoint(
    training_run_id=run["id"],
    checkpoint_number=1,
    checkpoint_file="path/to/checkpoint.pt"
)
```

### Edge Function API

Alle Edge Functions sind verfügbar unter:
```
https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/
```

#### Rollout erstellen
```bash
curl -X POST https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/submit-rollout \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -d '{"episode_number":1,"environment":"alfworld"}'
```

#### Metriken abrufen
```bash
curl https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/get-metrics?environment=alfworld
```

## 📦 Erstelle Dateien

### Datenbank
- ✅ `supabase/migrations/20241109_initial_schema.sql` - Hauptschema
- ✅ `supabase/migrations/20241109_storage_buckets.sql` - Storage-Konfiguration

### Edge Functions
- ✅ `supabase/functions/submit-rollout/` - Rollout-Submission
- ✅ `supabase/functions/log-agent-state/` - Agent-State-Logging
- ✅ `supabase/functions/complete-rollout/` - Rollout-Abschluss
- ✅ `supabase/functions/get-metrics/` - Metriken-API

### Python-Integration
- ✅ `openmanus_rl/utils/supabase_client.py` - Client-Setup
- ✅ `openmanus_rl/utils/supabase_db.py` - Datenbank-Operationen
- ✅ `openmanus_rl/utils/supabase_storage.py` - Storage-Verwaltung

### Deployment
- ✅ `.github/workflows/supabase-deploy.yml` - GitHub Actions
- ✅ `scripts/deploy_supabase.ps1` - Deployment-Script
- ✅ `scripts/upload_datasets.py` - Dataset-Upload

### Dokumentation
- ✅ `docs/SUPABASE_INTEGRATION.md` - Vollständige Dokumentation

## 🔐 GitHub Secrets

Für automatisches Deployment via GitHub Actions, fügen Sie diese Secrets hinzu:

1. Gehen Sie zu: `https://github.com/u4231458123-droid/nexifyai-openmanus/settings/secrets/actions`

2. Fügen Sie hinzu:
   - `SUPABASE_ACCESS_TOKEN`: Ihr Supabase Access Token
   - `SUPABASE_SERVICE_ROLE_KEY`: `sbp_ed71b8e9dd2c7d7205d626b99ad63a218934e67c`

## 📚 Nächste Schritte

1. **Datenbank-Schema anwenden**
   ```powershell
   supabase db push
   ```

2. **Edge Functions deployen**
   ```powershell
   supabase functions deploy --project-ref jdjhkmenfkmbaeaskkug
   ```

3. **Datasets hochladen**
   ```powershell
   python scripts/upload_datasets.py
   ```

4. **Training-Scripts aktualisieren**
   - Integrieren Sie Supabase-Logging in Ihre Trainings-Loops
   - Verwenden Sie `TrainingRunManager` und `RolloutManager`

5. **Monitoring einrichten**
   - Nutzen Sie das Supabase Dashboard: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug
   - Verwenden Sie die `/get-metrics` API für Live-Statistiken

## 🛠️ Troubleshooting

### "Permission denied" beim Push
Verwenden Sie HTTPS statt SSH:
```powershell
git remote set-url origin https://github.com/u4231458123-droid/nexifyai-openmanus.git
```

### "Unauthorized" Fehler
Überprüfen Sie Ihre API-Keys in `.env.supabase`

### Migrations-Fehler
```powershell
supabase db reset
supabase db push
```

## 📖 Weitere Dokumentation

Siehe `docs/SUPABASE_INTEGRATION.md` für:
- Detaillierte API-Referenz
- Datenbankschema-Übersicht
- Erweiterte Verwendungsbeispiele
- Best Practices

## ✅ Status

🟢 **Alle Komponenten sind konfiguriert und bereit!**

- [x] Supabase-Projekt verknüpft
- [x] Datenbank-Migrationen erstellt
- [x] Storage-Buckets konfiguriert
- [x] Edge Functions implementiert
- [x] Python-Client integriert
- [x] GitHub Actions eingerichtet
- [x] Dokumentation vollständig

**Das Projekt ist produktionsbereit und kann deployed werden!** 🚀
