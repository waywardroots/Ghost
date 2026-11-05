# Music Assistant - Phase 1: Library Indexer

Catalogs your samples, plugins and Ableton Live projects, and learns your style from them.

## Setup (Windows, one time)
1. Install Python 3.12 from python.org (tick "Add python.exe to PATH").
2. Put `indexer.py` and `requirements.txt` in a folder, e.g. `C:\MusicAssistant`.
3. Open PowerShell in that folder and run:
   ```
   pip install -r requirements.txt
   ```

## Run it
```
python indexer.py scan --samples "D:\Samples" "C:\Users\YOU\Documents\Splice" --projects "D:\Music\Ableton Projects"
```
- Pass every folder that holds samples and every folder that holds your `.als` projects.
- Plugins are found automatically in the standard VST2, VST3 and CLAP folders. Add others with `--plugins "E:\Plugins"`.
- The first scan analyzes every sample (BPM, key, loop or one-shot) and can take a while on a big library. Later scans only touch new or changed files.
- In a hurry? Add `--no-audio-analysis` to index by file and folder names only.

## Find samples
```
python indexer.py search --category kick
python indexer.py search --key Am --loops
python indexer.py search dark --bpm 140
python indexer.py search --category 808 --key "F# minor"
```

## See your style profile
```
python indexer.py profile
```
Shows your usual tempos and keys, favourite plugins and Live devices (and which track types you use them on), and your go-to samples.

## What it creates
In `C:\Users\YOU\MusicAssistant\`:
- `music_library.db` - the catalog the assistant will search in later phases
- `style_profile.json` - a summary of how you make music

Nothing is uploaded anywhere; everything stays on your PC. Your projects and samples are only read, never changed.

## Notes
- Backup copies of Live sets (in `Backup` folders) are skipped.
- BPM and key from file names win over detection. Detected keys are skipped for drums.
- A project's key is estimated from its MIDI notes, ignoring drum tracks.
