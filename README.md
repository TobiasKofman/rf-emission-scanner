# drone-emission

Passive RF monitor for the 2.4 GHz band. An ADALM-Pluto captures raw I/Q, a Raspberry Pi 5
looks for unexpected transmitters in it, and detections get sent over serial to an ESP dev
kit that drives an OLED, LED and buzzer.

Right now we're focused on drones. Most RC control links use frequency hopping (FHSS), so in
a spectrogram they show up as short narrow bursts jumping all over the 2400-2483.5 MHz band,
which is pretty easy to pick out.

Hardware chain: antenna, Pluto, USB to the Pi, serial to the ESP.

## Events

The Pi sends one JSON object per line to the ESP, like this:

```json
{"type":"detection","ts":"2026-11-13T14:32:05Z","alert":"critical","label":"FHSS/control-link-like","confidence":0.82,"rssi_dbm":-58.0,"center_mhz":2440.0,"bw_mhz":2.0,"duration_ms":100.0}
```

- `alert` is info, warning or critical. Only warning and up get sent by default.
- `label` is one of `FHSS/control-link-like`, `wideband/video-downlink-like`, `BT-like` or
  `WiFi-like`. The last two are info level, so they aren't sent by default.
- `confidence` is 0 to 1.
- `rssi_dbm` is the median burst power (dBFS minus RX gain plus `rssi_cal_db`). Not calibrated
  yet, so only useful as a relative number for now.
- `center_mhz` is the middle of the frequency range the bursts covered.
- `bw_mhz` is the median bandwidth of a single burst.
- `duration_ms` is the time from the first burst to the end of the last one in the window.

## How it works

The Pluto only sees 20 MHz at a time, so each update sweeps five LOs (2410 to 2474 MHz in
16 MHz steps) to cover 2400-2484 MHz. Each stop is a 65536 sample buffer at 20 MSPS (about
3.3 ms), and one buffer gets thrown away after each retune while the LO settles. That means
any one channel is only watched about 10% of the time or less.

For each buffer we:

1. Compute a spectrogram (256 point FFT, hop of 128).
2. Estimate the noise floor and smooth it across buffers. An LO can't sit more than 3 dB
   above the quietest one, so a signal filling a whole capture doesn't become its own floor.
3. Threshold it and group connected cells into bursts, ignoring anything more than 35 dB
   below the frame's peak since that's FFT leakage. Each burst gets a start time, duration,
   center freq, power and bandwidth (the part holding 90% of its power).
4. Check wide bursts for the 3.2 us OFDM symbol that identifies Wi-Fi.

After each full sweep, we look at a rolling 1 second window and count narrow bursts, how many
different channels they land on, the hop rate, median bandwidth/duration, and how much of the
time wide signals are present. Narrow bursts inside a wide one are pieces of it, so they
aren't counted. The rules in `rfsense/classify.py` turn that into a label:

- narrow bursts on at least 8 channels: FHSS/control-link-like
- same thing but with bursts under ~0.5 ms: BT-like (info only, not sent)
- a non Wi-Fi wide signal (8 MHz+) on air at least 20% of the time: wideband/video-downlink-like
- a Wi-Fi signal on air at least 20% of the time: WiFi-like (info only, not sent)

To avoid spamming the ESP, a label has to show up 3 sweeps in a row before the first event
is sent. After that it's resent once a second, or right away if the alert level goes up.

The wideband and Wi-Fi thresholds are tuned on DroneRF (see Tuning). The FHSS and Bluetooth
ones are still guesses because that dataset doesn't contain a measurable hopping link.

## Code layout

- `config/default.yaml` has every setting
- `rfsense/sources.py` reads from the Pluto, a recorded file, or a simulator
- `rfsense/dsp.py` does the spectrogram and noise floor
- `rfsense/detect.py` finds bursts
- `rfsense/classify.py` computes the window features and labels
- `rfsense/events.py` defines the JSON event
- `rfsense/sinks.py` handles output (stdout, serial, udp, file)
- `rfsense/pipeline.py` is the main loop
- `tools/capture.py` records raw IQ to a file
- `tools/plot_capture.py` plots a recording with the detected bursts boxed
- `tools/dronerf.py` downloads the DroneRF dataset and replays it through the detector
- `tests/` runs everything on simulated signals

## Setup

### On a laptop (no hardware)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python -m rfsense --scenario fhss --sinks stdout
```

Other sim scenarios: noise, bluetooth, wideband, mixed.

### On the Pi

```bash
sudo apt update
sudo apt install -y python3-venv libiio-utils python3-libiio
iio_info -s  # the Pluto should show up here

# --system-site-packages so the venv can see python3-libiio from apt
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[sdr,dev]"

# so we can open the ESP serial port (log out and back in after)
sudo usermod -aG dialout $USER
```

Copy `config/default.yaml` to `config/local.yaml` and set the Pluto uri, the serial port
(check `ls /dev/ttyUSB* /dev/ttyACM*`) and the gain. local.yaml is gitignored and gets
used automatically if it exists.

```bash
python -m rfsense  # Pluto to ESP
python -m rfsense --sinks stdout  # just print events
```

Every 5 seconds it logs how long processing takes per buffer compared to the buffer length,
plus the noise floor and current features. If processing time is longer than the buffer
time, the Pi isn't keeping up.

## Tuning

Record some data with and without the drone on:

```bash
python tools/capture.py -n 100 -o captures/baseline.npy
python tools/capture.py -n 100 -o captures/drone_on.npy
```

Plot a buffer (saves a png, open it in VS Code):

```bash
python tools/plot_capture.py captures/drone_on.npy --index 10
```

Then replay it through the detector while changing settings in local.yaml:

```bash
python -m rfsense --file captures/drone_on.npy --sinks stdout
```

capture.py prints a warning if the signal is close to clipping, in which case lower
`gain_db`. You can also set `report.save_iq_dir` to save the IQ behind every event that
gets sent, which is handy for building up labeled examples.

### DroneRF

[DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1) has 227 recordings of a Bebop, an
AR Drone, a Phantom and background RF. This replays them with the same sweep timing as the
Pluto:

```bash
python tools/dronerf.py fetch  # about 4 GB download, 9 GB converted, needs bsdtar
python tools/dronerf.py eval
python tools/dronerf.py eval --atten 0 10 20 30  # extra path loss in dB
```

Tune on the default split and keep `--split holdout` for a final check. What it showed:

- The Bebop and AR are plain Wi-Fi (about 95% of their wide bursts have the OFDM fingerprint),
  so they get `WiFi-like` and no alert. Telling them from a router needs a site baseline.
- The Phantom's wide signal isn't Wi-Fi OFDM, which is what the wideband rule keys on.
- About 90% of the narrow "bursts" near wide signals were fragments of them, not hops.
- Signal strength separates drones from background perfectly here, but only because the
  background recordings are nearly silent. Don't use it as a feature.

The recordings are 125 ms each from one site, with three drones from around 2014-2016 and no
ELRS, FrSky or newer DJI links, so treat the results as a sanity check.

## TODO / known issues

- RSSI isn't calibrated. Need to measure `rssi_cal_db` against a known source.
- Bluetooth hops too. Right now we only use burst length to tell it apart from RC links,
  which needs checking against real data (BT audio uses longer packets).
- The FHSS rule hasn't been tested on a real hopping link yet.
- Signals between 5 and 8 MHz wide count as neither narrow nor wide, so they're ignored.
- No baseline/anomaly model yet, so "unexpected" just means "looks like a drone" for now. A per-site baseline which learns the normal Wi-Fi at the target location is necessary to detect Wi-Fi drones.