# Ghost

A personal AI music assistant for Ableton Live that knows your samples, your plugins and your style.

| Phase | What it does | Status |
|---|---|---|
| 1. Library indexer | Catalogs samples, plugins and Live projects; learns your style | Done |
| 2. Ableton control | Lets the assistant build in Live: tracks, clips, notes, plugins, samples | Done |
| 3. Your-style MIDI model | Writes melodies, chords and drums that sound like you | Next |
| 4. Your-style audio model | Generates loops and textures in your sound | Later |

## Setup (Windows, one time)
1. Install Python 3.12 from python.org (tick "Add python.exe to PATH").
2. Clone or download this repo, e.g. to `C:\Ghost`.
3. Open PowerShell in that folder and run:
   ```
   pip install -r requirements.txt
   ```

---

## Phase 1: Library indexer

### Run it
```
python indexer.py scan --samples "D:\Samples" "C:\Users\YOU\Documents\Splice" --projects "D:\Music\Ableton Projects"
```
- Pass every folder that holds samples and every folder that holds your `.als` projects.
- Plugins are found automatically in the standard VST2, VST3 and CLAP folders. Add others with `--plugins "E:\Plugins"`.
- The first scan analyzes every sample (BPM, key, loop or one-shot) and can take a while on a big library. Later scans only touch new or changed files.
- In a hurry? Add `--no-audio-analysis` to index by file and folder names only.

### Find samples
```
python indexer.py search --category kick
python indexer.py search --key Am --loops
python indexer.py search dark --bpm 140
```

### See your style profile
```
python indexer.py profile
```

Creates `music_library.db` and `style_profile.json` in `C:\Users\YOU\MusicAssistant\`. Nothing is uploaded; your projects and samples are only read, never changed.

---

## Phase 2: Ableton control

Two parts work together:
- **Ghost Remote Script** (`ableton/Ghost`) runs inside Live and carries out commands. It only accepts connections from your own PC (127.0.0.1, port 9877).
- **Ghost connector** (`mcp_server.py`) gives the assistant its tools: building in Live, plus searching your library from Phase 1.

### Install
```
python install.py
```
This copies the Remote Script into `Documents\Ableton\User Library\Remote Scripts\Ghost` and adds Ghost to Claude Desktop's connectors (your existing Claude config is backed up first).

Then:
1. Restart Live. In **Settings > Link, Tempo & MIDI**, set a **Control Surface** to **Ghost** (Input and Output: None). Live shows "Ghost assistant ready".
2. Fully quit and reopen Claude Desktop.
3. Check the link: `python mcp_server.py --check`

### Try it
Ask Claude things like:
- "Look at my style profile, then start a new idea at my usual tempo: drums with my favorite kit, a bassline in my most-used key, and pads on Serum."
- "Find a dark 140 BPM vocal loop in my samples and put it on a new audio track."
- "Read the melody in Bass clip 1 and write a counter-melody on a new track."
- "Put an EQ Eight and Pro-Q 3 on the Vox track and set the low cut higher."

### Tools the assistant gets
Session: get_session_info, get_track_info, set_tempo, set_time_signature, start/stop_playback, undo.
Tracks: create_midi_track, create_audio_track, set_track_name, set_track_volume, delete_track.
Clips and notes: create_clip, add_notes, get_notes, set_clip_name, set_clip_loop, delete_clip, fire_clip, stop_track_clips, fire_scene.
Sound: search_live_browser, load_device (Live devices, VST/AU plugins, presets, drum kits), load_sample, get/set_device_parameter.
Your library: search_samples, list_my_plugins, get_style_profile.

### Good to know
- Samples load straight from any folder on Live 12.0.5+ (onto audio tracks). Otherwise, add your sample folders to **Places** in Live's browser so Ghost can find them.
- Plugins must be enabled in Live (Settings > Plug-Ins) to appear in its browser.
- Everything Ghost does can be undone with Ctrl+Z in Live.

### Test without Live
```
python tests/test_phase2.py
```
Runs the full chain against a simulated Live (36 checks).
