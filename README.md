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
Sound: search_live_browser, load_device (Live devices, VST/AU plugins, presets, drum kits), load_sample.
Listening: listen_to_track, analyze_audio_file.
Arrangement: build_arrangement (Session clips onto the timeline, with named locators), get_arrangement_info, set_locators, set_scene_name, duplicate_clip, place_clip (variations/fills), place_audio_clip (risers, impacts), automate (filter sweeps and other automation in real units), get_drum_pads, clear_arrangement, read_project_arrangement (map the structure of any of your .als files to model a new song on it).
Sound design: get_device_parameters, set_device_parameters, set_device_parameter, save/recall/list_sound_snapshot(s), find/read/write_preset_file.
Your library: search_samples, list_my_plugins, get_style_profile.

### Sound design
Ask for sounds in plain words: *"Load Serum on a new track and make a dark detuned reese bass"*, *"give me three pad variations on Pigments and let me switch between them"*, *"make my Diva lead brighter with a slower attack"*.

- **Any plugin or Live device:** Ghost reads its controls and sets them in real units (`800 Hz`, `35 ms`, `-6 dB`, `Saw`). Many settings can change in one step.
- **Snapshots:** Ghost saves versions of a sound (A, B, C...) so you can compare them and go back. They're kept in `MusicAssistant\snapshots.json`.
- **Serum 2 and u-he Diva, Hive, Repro-1, Repro-5:** Ghost can write complete presets as files. It starts from a factory/init patch or one of yours and saves the result to a `Ghost` folder in that synth's preset browser (Serum 2: `Presets\User\Ghost`; u-he: `UserPresets\<Synth>\Ghost`). Originals are never changed. Serum 2 only stores settings that differ from default, so Ghost can change what a preset already uses, plus settings that a matching part of the preset uses (e.g. detune on oscillator B when oscillator A has it).
- **Big plugins** (Serum 2 live tweaking, Pigments, Omnisphere, Spire...) only share the controls you expose. Do it once per plugin: open it in Live, click **Configure** (wrench icon), move the knobs you want Ghost to use (up to 128), then click Configure again. Saving the plugin as a Live preset afterwards keeps that setup.
- To keep a sound from other plugins, save it in the plugin or with Live's Save Preset button. Ghost can't write their preset files.

### Listening
Ghost can check its own work. `listen_to_track` plays a phrase on a track (soloed), records it inside Live through **Resampling**, and measures the result: low end per band, brightness, pitch, grit, movement (e.g. reese beating, wobble rate), stereo width (and whether the low end is too wide), attack and sustain. It also renders a spectrogram picture Claude can look at. `analyze_audio_file` does the same for any sample or reference, and `compare_to` lists how two sounds differ, so Ghost can shape a sound until it matches a reference.

Recordings are copied to `MusicAssistant\listening\`. Listening stops Live's playback while it records, and puts your solo settings back afterwards. Check it yourself any time: `py -3.13 listen.py sound.wav [reference.wav]`.

### Good to know
- Samples load straight from any folder on Live 12.0.5+ (onto audio tracks). Otherwise, add your sample folders to **Places** in Live's browser so Ghost can find them.
- Plugins must be enabled in Live (Settings > Plug-Ins) to appear in its browser.
- Everything Ghost does can be undone with Ctrl+Z in Live.

### Test without Live
```
python tests/test_phase2.py
python tests/test_presets.py
```
Runs the full chain against a simulated Live, plus preset-file tests (145 checks).
