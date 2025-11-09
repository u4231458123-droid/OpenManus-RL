# OpenManus-RL Supabase & MCP Integration

Dieses Dokument beschreibt die vollständige Einrichtung und Erweiterung des Projekts nach Originalstruktur (manus.mi), inkl. Supabase, MCP-Server und Open-Source-Integrationen.

## 1. Supabase-Konfiguration

- `.env.supabase` wie im Original (siehe Datei)
- Datenbank-Passwort eintragen
- Supabase CLI installieren: `npm install -g supabase`
- Projekt verlinken: `supabase link --project-ref jdjhkmenfkmbaeaskkug`
- Migrationen ausführen: `supabase db push`
- Edge Functions deployen:
  ```bash
  supabase functions deploy submit-rollout
  supabase functions deploy log-agent-state
  supabase functions deploy complete-rollout
  supabase functions deploy get-metrics
  ```

## 2. Python-Abhängigkeiten

- Supabase-Python-Bibliothek: `pip install supabase`
- Weitere Requirements: `pip install -r requirements.txt`

## 3. MCP-Server-Integration

- MCP-Server als Backend für Agent-Kommunikation und Tool-Calls
- Open-Source-Frameworks wie `verl`, `AgentGym`, `TinyZero`, `OpenR1`, `Trlx` sind integriert
- Erweiterbar um weitere Open-Source-Tools (z.B. HuggingFace, LangChain, Toolformer)

## 4. Beispiel für Supabase-Logging

```python
from openmanus_rl.utils.supabase_client import get_supabase
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager
from openmanus_rl.utils.supabase_storage import StorageManager

run = TrainingRunManager.create_run(
    name="alfworld-gigpo-experiment",
    algorithm="gigpo",
    environment="alfworld",
    config={"learning_rate": 0.0001, "batch_size": 32}
)
rollout = RolloutManager.create_rollout(
    training_run_id=run["id"],
    episode_number=1,
    environment="alfworld",
    task_description="Find the red ball"
)
# ... weitere Logging- und Storage-Operationen ...
```

## 5. Erweiterungen & Open-Source

- Integration weiterer Open-Source-Agenten, Tool-APIs und Benchmarks möglich
- Dokumentation und Beispiele in `docs/` und `README.md`
- Eigene Umgebungen und Belohnungsfunktionen können nach Guide erweitert werden

## 6. Test & Validierung

- Trainingslauf starten und Supabase-Daten prüfen
- Storage-Buckets und Edge Functions testen
- Logs und Checkpoints hochladen

## 7. Weiterführende Dokumentation

- `docs/DEVELOPMENT_GUIDE_EN.md` und `docs/SUPABASE_INTEGRATION.md`
- Original-Frameworks: [Verl](https://github.com/volcengine/verl), [AgentGym](https://github.com/THUDM/AgentGym)

---

**Hinweis:**
Dieses Setup ist vollständig kompatibel mit dem Original und kann beliebig erweitert werden. Für weitere Integrationen oder Automatisierungen einfach melden!
