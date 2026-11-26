# RedEye concept: the T-800 view, formalised

RedEye recreates the *behaviour and visual grammar* of the T-800's
point-of-view shots in The Terminator (1984, "T1") and Terminator 2 (1991,
"T2"), as a mixed-reality layer over the real world. We reproduce mechanics,
not footage: no film frames, logos or copied text ship with the app, and all
on-screen strings are our own.

Research notes, corrections to popular memory, and sources are at the end.

## Production facts that shape the design

- **T1** ("Termovision", Ernest Farino / Kinetic Image) was an infrared-like
  red view with **real Apple II 6502 listings** from *Nibble* magazine
  (plus COBOL and *73 Magazine* excerpts) scrolling in white. The script asks
  for a view that is "bright and hyper-real… figures more luminous than the
  background", with data that "changes more rapidly than any human eye could
  follow". The data is decoration that implies computation; nobody reads it.
- **T2** (concept Van Ling, made by Video Image) scanned film to ~1 MP
  greyscale, remapped it to black/white/red, rotoscoped subject outlines,
  and faked "enhance" by playing a mosaic backwards. Same idea, much more
  structured: labelled panels, outlines, insets.

## Mechanics

Ratings are for real-time, on-device work on Quest 3. "Needs" lists the
perception a real system requires. M-numbers match [`hud::Mechanic`](../src/hud/mod.rs).

| # | Mechanic | Film | Needs | Rating | Milestone |
|---|---|---|---|---|---|
| M1 | Colour grade | T1, T2 | — | easy | M1 |
| M2 | Data scroll | T1, T2 | — (or real telemetry) | easy | M2 |
| M3 | Target lock | T1, T2 | person detection, tracking, prediction | medium | M3 |
| M4 | Acquisition inset | T2 | face detection, crop | medium | M4 |
| M5 | Size assessment / MATCH | T2 | pose, metric depth, floor | medium | M4 |
| M6 | Object identification | T1, T2 | detection, mask, crop classifier | easy–hard | M4 |
| M7 | Threat assessment | T2 | multi-person tracking | medium | M5 |
| M8 | Response menu | T1 | voice activity, speech-to-text | easy–medium | M5 |
| M9 | Text reading | T1 | OCR | medium | M5 |
| M10 | Expression analysis | T2 SE | face landmarks | medium | later |
| M11 | Night vision / enhance | T2 | head yaw | easy | M2 |
| M12 | Boot / diagnostic | T2 SE | — | easy | M2 |
| M13 | Damage / power failure | T2 | — (battery, hand impacts) | easy | M2 |
| M14 | Audio analysis | (implied) | mic level, pitch | easy | later |

### M1 Colour grade

Luminance mapped to a red ramp. **T1**: about 45% of pixels are crushed to
black, midtones are pure red (G ≈ 0.14 R, B ≈ 0.10 R), highlights clip to
pink-white, heavy grain. **T2**: lifted blacks (about 0.3% pure black),
lower contrast, magenta-pink highlights around (0.96, 0.25, 0.28).

Model (see [`grade.rs`](../src/grade.rs)): levels → gamma → lift; R = y,
G = gk·y^gp, B = bk·y^bp; blend to a hot colour past a clip point; grain is
added to luminance *before* grading; vignette and scanlines are applied
after. Presets fitted to web stills:

| Preset | black | white | gamma | lift | gk / gp | bk / bp | clip | grain | vignette | scanlines |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | .12 | .75 | 1.3 | 0 | .14 / 1 | .10 / 1 | .92 | .08 | .25 | 0 |
| T2 | .02 | .92 | .9 | .08 | .25 / 2 | .30 / 2.2 | .95 | .03 | .12 | 0 |
| Damaged | T2 | | | | | | | .12 | .40 | .45 |

The tone curve (everything except grain, vignette and scanlines) becomes the
passthrough colour map, so the compositor grades the world for free. The
spatial effects go on our own layer as overlays: additive grain, plus
black-with-alpha vignette and scanlines.

### M2 Data scroll

**T1**: small white monospace code and hex, flickering in the lower centre
and upper left. **T2**: labelled blocks ("ANALYSIS:", "TRACKING:", "CODE:")
with a dotted rule over 3–6 rows of digit columns, in the corners. They
repeat across scenes, so they are filler. *Trigger*: always on, with bursts
on events. *Ours*: generated pseudo-assembly and real telemetry (frame time,
model latency, track count, battery, head pose) rendered as number columns.
That is honest data, and in the film's spirit.

### M3 Target lock

A centred disc with a thin crosshair: translucent in T1, two-tone in T2. It
follows a moving subject. The T1 script also describes a ghost outline that
extrapolates a subject's motion behind cover (not confirmed in the cut).
*Ours*: detection → tracking ID → world-space position from depth →
constant-velocity prediction. That prediction is needed anyway, to hide 60–100 ms of
camera-to-display latency. Lock acquisition animates from search → acquiring →
locked, and only confirmed tracks are eligible.

### M4 Acquisition inset

The view snap-zooms to a pixelated freeze-frame of the face, boxed at top
centre, stamped "TARGET ACQUIRED / IDENT POSITIVE ■". A right-hand column
lists "MATCH CRITERIA", a name, attribute rows (HGHT, WGHT, HAIR, EYES…) and
a percentage match. *Ours*: a face crop from the camera frame, mosaic-resolved
(the T2 "enhance" trick, run forwards), with attributes we can actually
measure (height, distance, speed). **No face recognition of strangers.** A
name appears only for users who enrol themselves and consent.

### M5 Size assessment / MATCH (T2 biker bar)

A status stack on the right ("SCAN MODE… SIZE ASSESSMENT… FIT PROBABILITY"),
a number block on the left, and a large "MATCH" at bottom centre. A scan of
the building with a grid box comes first. *Ours*: 2D keypoints plus metric
depth plus the floor plane give height, shoulder width and inseam (hip to
floor minus an empirical offset), compared against the wearer's own height
from tracking. The Depth API is low resolution, so expect errors of a few
centimetres. Present the numbers with false precision, as the film does.

### M6 Object identification

**T2**: a solid and dashed outline traced around a motorcycle, a "VISUAL:"
model line, and a spec table (VEHI, SIZE, TSPD…). **T1**: a boxed schematic
with lettered callouts. *Ours*: class from the detector (easy), outline
from a mask (medium), finer identity from zero-shot crop classification
against a fixed vocabulary (hard). Spec tables come from a bundled
per-class table.

### M7 Threat assessment

Corner blocks "TRAJECTORY LOGGING" and "PARAMETERS", "THREAT ASSESSMENT:
POTENTIAL DAMAGE", "TERMINATION OVERRIDE / DISABLE TARGETS ONLY", and
"HUMAN CASUALTIES: 0.0". *Ours*: people count, tracks and trajectories. **We
never label a real person as a threat**: the panel reports counts and motion,
and casualties are always 0.0, as in the film.

### M8 Response menu (T1 hotel room)

"POSSIBLE RESPONSE:" with six left-aligned options in large white type at
centre screen. A dash cursor steps to the chosen option, which is then
spoken. *Ours*: voice activity opens the menu; the options are canned or
matched to a transcript; the user picks with a hand pinch or a gaze dwell.
Easy with canned options, medium with live speech-to-text.

### M9 Text reading (T1 address book)

Handwriting is transcribed beneath itself in large white type and updates as
pages turn. *Ours*: printed signs and labels via OCR on a region the user
looks at, on demand. Handwriting is out of scope.

### M10 Expression analysis (T2 Special Edition)

"MAGNIFICATION:" and a black inset with a cyan triangulated mouth mesh.
Needs a face-landmark model and a short frame buffer. Deferred.

### M11 Night vision / image enhance

"TRACKING:" top left, an eight-point compass rose top right, and "PSD MODE /
IMAGE ENHANCE ■" with a mosaic resolving to a sharp image. Quest exposes no
compass, so the rose is driven by tracked yaw relative to the session start.

### M12 Boot / diagnostic

From the T2 Special Edition chip scene: static, black, then a model/version
block, time-gap and power checks with number tables, then "DIAGNOSTIC ■".
*Ours*: plays on launch, with our own strings and real self-tests (camera
opened, models loaded, GPU timing).

### M13 Damage / power failure (T2 steel mill)

A flat red field filled wall to wall with dithered data, ending "IMMINENT
SHUTDOWN", then a circuit panel with a bright path routed to a battery icon:
"REROUTE", "ALTERNATE POWER ■". Scanlines, cyan fringing, jitter and flicker
throughout. *Ours*: the damaged grade plus temporal artefacts (1–3 px channel
split, row jitter, 8–12 Hz flicker), triggered by low battery, a thermal
warning, or a hand punching the HUD.

### M14 Audio analysis

The films show no audio HUD beyond "SPEECH LEVEL 78" in the bar scene.
*Ours*: microphone level and pitch as number blocks. Sound direction and voice
imitation are out of scope (no raw mic-array access, and a consent problem).

## Visual grammar

- **Colours**: the grade, white text, and a little cyan for schematics and
  meshes (T2). Overlays are drawn *in* the red world: white text is a
  readable highlight, not UI chrome.
- **Typography**: T1 display text is Bessellen (VGC, 1972), which has no
  official digital release. T2 uses a crude redraw of it plus Helvetica, and
  the code listings look daisy-wheel printed (Letter Gothic-like). Free
  candidates:
  - **Modern Vision**, a Bessellen reconstruction (CC BY-ND 3.0: requires
    attribution and forbids modification, so check whether an SDF atlas
    counts as modification before shipping it);
  - **TeX Gyre Heros** (GUST) or **Archivo**, expanded (OFL), for Helvetica;
  - **Share Tech Mono**, **Terminus** (OFL), or VileR's Oldschool PC fonts
    (CC BY-SA) for code.
- **Motion**: text is typed out, never popped in ([`hud::text`](../src/hud/text.rs));
  a solid block cursor ■ marks the current line; numbers churn; insets snap.
- **Layout (T2)**: number blocks in the corners, a status stack mid-right,
  one state word bottom-centre.
- **Comfort**: the HUD lives on a body-locked canvas 1.5–2 m away that lazily
  follows the head, with text within about ±25° of centre. Target markers are
  world-locked to the targets. The grade and grain are world-anchored via the
  compositor and never strobe; damage flicker is brief and opt-out.

## Ethics and privacy

- Everything runs on the device. No frames, audio or derived data leave it.
- No identification of strangers. Faces are detected, never recognised,
  except for self-enrolled, consenting users.
- No real person is labelled a threat or a target for violence. Lock-on is a
  visual effect over "subject", and casualty counters read zero.
- The camera is released whenever the app loses focus.

## Corrections to popular memory

- The response menu is in the T1 **hotel room** (the janitor's "dead cat?"),
  not the punk scene; T1's arrival and punk scenes have no HUD.
- "MATCH" on clothes, boots and motorcycle is **T2's biker bar**.
- The Cyberdyne boot screen is the **T2 Special Edition CPU-chip** scene. The
  impalement sequence is "IMMINENT SHUTDOWN" → "REROUTE" → "ALTERNATE POWER".
- No HUD is shown for the T-1000, the voice-imitation calls, weapon ID (that
  is T3), or T1's eye repair. The "DECIBEL OVERLOAD" still likely comes from
  the "You Could Be Mine" music video. Fan-made frames circulate, so verify
  stills against a disc before citing them.

## Sources

- Terminator Wiki, Head-up display and gallery:
  <https://terminator.fandom.com/wiki/Head-up_display>,
  <https://terminator.fandom.com/wiki/Head-up_display/Gallery>.
  Frame files `T-800_Responses.jpg`, `T-800_Pursuit.jpg`,
  `T-800_Handwriting.jpg`, `T-800a_Clothing.jpg`, `T-800a_Positive_ID.jpg`,
  `T-800a_Threat.jpg`, `T-800a_Casualities.jpg`, `T-800a_Night_Vision.jpg`,
  `T-800Smileanalisis.png`, `T-800a_Bootup.jpg`,
  `T2-t800-film-hud-shutdown.jpg`, `T2-t800-film-hud-rerouting.jpg`.
- The 6502 code: <https://www.pagetable.com/?p=64>,
  <https://hackaday.com/2024/04/15/analyzing-the-code-from-the-terminators-hud/>,
  <https://nofilmschool.com/2017/08/watch-meaning-terminators-heads-display-revealed>,
  <https://www.marksimonson.com/notebook/view/TheAppleTerminatorII>.
- Production: <https://www.sideshow.com/blog/how-terminator-2-achieved-iconic-special-effects>,
  <https://www.hudsandguis.com/home/2011/01/02/terminator-the-evolution-of-machine-vision>.
- Scripts: <https://imsdb.com/scripts/Terminator.html>,
  <https://imsdb.com/scripts/Terminator-2-Judgement-Day.html>.
- Fonts: <https://fontsinuse.com/typefaces/170077/bessellen>,
  <https://fontsinuse.com/uses/46291/terminator-2-judgment-day-1991-movie-logo-and>,
  <https://fontstruct.com/fontstructions/show/477683>.
- Prior art: <https://blogs.windows.com/windowsdeveloper/2017/03/06/building-terminator-vision-hud-hololens/>,
  Meta's passthrough colour mapping and camera object-detection samples.
