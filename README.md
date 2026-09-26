# AI Video Intelligence Studio

This Streamlit application analyzes uploaded video clips and reports:

- Scene boundaries and timestamps
- People counts and tracked identities
- Detected objects and instruments
- Local event categories and actions
- Optional vision-language descriptions, including appearance details
- A playable subclip for any requested time range

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app\web\streamlit_app.py
```

Streamlit opens the app in your default browser when it starts. It may still
print the local and network addresses in the terminal; those are informational.

The local YOLO and YOLO-World analysis works without an API key.

## Optional vision-language enrichment

The app supports any OpenAI-compatible multimodal endpoint. Configure it
through environment variables before starting Streamlit:

```powershell
$env:OPENAI_VISION_API_KEY = "your-key"
$env:OPENAI_VISION_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_VISION_MODEL = "gpt-4o-mini"
```

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are also accepted as
fallback names. Provider failures are surfaced in the application; local
analysis should be used when no provider is configured.

With vision enrichment enabled, each scene is also checked against the
grouped scene taxonomy in `app/intelligence/scene_taxonomy.py`. Returned
labels include confidence and visual/timeline evidence and are restricted to
that taxonomy. Cinematic or emotional labels that cannot be supported by the
available frame and timeline context are omitted; without a configured vision
provider, the local detector continues to report only its supported events.

## Time-range questions

After analysis, enter start and end times as minutes and seconds. The
application returns the analyzed scenes overlapping that range and uses the
selected interval for video playback. For example, enter 10 minutes and 30
seconds as `10:30`.