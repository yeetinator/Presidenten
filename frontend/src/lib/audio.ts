import { get } from "svelte/store";
import { persisted } from "./persisted";

export type SoundName =
  | "deal"
  | "shuffle"
  | "cardPlay"
  | "quad"
  | "pass"
  | "pileClear"
  | "cardSwap"
  | "playerOut"
  | "playerOutOwn"
  | "roundEnd"
  | "jumpIn";

type Channel = "effects" | "alerts";

// Schedules a synthesized sound into `out` from `at`; returns its length in seconds
type Synth = (ctx: AudioContext, out: AudioNode, at: number) => number;

interface SoundDef {
  channel: Channel;
  gain: number;
  // An effect asked for again within this long is dropped
  cooldownMs: number;
  // Sample variants, one picked at random per play; a synth given as well is
  // layered on top
  src?: string[];
  synth?: Synth;
}

export interface SoundSettings {
  muted: boolean;
  effectsVolume: number;
  alertsVolume: number;
}

export const soundSettings = persisted<SoundSettings>("president.sound.v1", {
  muted: false,
  effectsVolume: 0.6,
  alertsVolume: 0.8,
});

const MAX_VOICES = 6;
const FAST_FORWARD_DUCK = 0.35;
const VOLUME_RAMP_S = 0.03;

function samples(name: string, variants: number[]) {
  return variants.map((variant) => `/sounds/${name}-${variant}.mp3`);
}

function tone(
  ctx: AudioContext,
  out: AudioNode,
  { type, freq, endFreq, at, duration, peak }: {
    type: OscillatorType;
    freq: number;
    endFreq?: number;
    at: number;
    duration: number;
    peak: number;
  },
) {
  const osc = ctx.createOscillator();
  const envelope = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, at);
  if (endFreq) osc.frequency.exponentialRampToValueAtTime(endFreq, at + duration);
  envelope.gain.setValueAtTime(0, at);
  envelope.gain.linearRampToValueAtTime(peak, at + 0.005);
  envelope.gain.exponentialRampToValueAtTime(0.0001, at + duration);
  osc.connect(envelope).connect(out);
  osc.start(at);
  osc.stop(at + duration + 0.02);
}

const knock: Synth = (ctx, out, at) => {
  tone(ctx, out, { type: "sine", freq: 170, endFreq: 70, at, duration: 0.09, peak: 0.8 });
  return 0.1;
};

// A deep thud under a card place, so finishing a quad lands like a slam
const slam: Synth = (ctx, out, at) => {
  tone(ctx, out, { type: "sine", freq: 120, endFreq: 45, at, duration: 0.2, peak: 0.9 });
  return 0.22;
};

function arpeggio(freqs: number[], step: number, duration: number, peak: number): Synth {
  return (ctx, out, at) => {
    freqs.forEach((freq, index) =>
      tone(ctx, out, { type: "triangle", freq, at: at + index * step, duration, peak }),
    );
    return (freqs.length - 1) * step + duration;
  };
}

// Bright and rising, unlike any card sound, so it is noticed straight away
const jumpInAlert: Synth = (ctx, out, at) => {
  [987.77, 1318.51].forEach((freq, index) => {
    const noteAt = at + index * 0.12;
    tone(ctx, out, { type: "triangle", freq, at: noteAt, duration: 0.16, peak: 0.6 });
    tone(ctx, out, { type: "square", freq, at: noteAt, duration: 0.16, peak: 0.1 });
  });
  return 0.3;
};

const SOUNDS: Record<SoundName, SoundDef> = {
  deal: { channel: "effects", gain: 0.3, cooldownMs: 110, src: samples("card-slide", [1, 2, 3, 4]) },
  shuffle: { channel: "effects", gain: 0.5, cooldownMs: 1000, src: ["/sounds/card-shuffle.mp3"] },
  cardPlay: { channel: "effects", gain: 0.9, cooldownMs: 120, src: samples("card-place", [1, 2, 3, 4]) },
  quad: { channel: "effects", gain: 1, cooldownMs: 120, src: samples("card-place", [1, 2, 3, 4]), synth: slam },
  pass: { channel: "effects", gain: 0.6, cooldownMs: 120, synth: knock },
  pileClear: { channel: "effects", gain: 0.7, cooldownMs: 120, src: samples("card-shove", [1, 2, 3, 4]) },
  cardSwap: { channel: "effects", gain: 0.7, cooldownMs: 120, src: samples("card-slide", [5, 6, 7, 8]) },
  playerOut: { channel: "effects", gain: 0.5, cooldownMs: 300, synth: arpeggio([523.25, 659.25, 783.99], 0.07, 0.25, 0.35) },
  playerOutOwn: { channel: "effects", gain: 0.7, cooldownMs: 300, synth: arpeggio([523.25, 659.25, 783.99, 1046.5], 0.08, 0.35, 0.35) },
  roundEnd: { channel: "effects", gain: 0.6, cooldownMs: 1000, synth: arpeggio([392, 523.25, 659.25, 783.99], 0.1, 0.9, 0.3) },
  jumpIn: { channel: "alerts", gain: 1, cooldownMs: 0, synth: jumpInAlert },
};

let ctx: AudioContext | null = null;
let effectsBus: GainNode | null = null;
let alertsBus: GainNode | null = null;
let effectsDucked = false;
let activeVoices = 0;
const buffers = new Map<string, AudioBuffer>();
const lastPlayed = new Map<SoundName, number>();

function createContext() {
  try {
    ctx = new AudioContext({ latencyHint: "interactive" });
  } catch {
    return; // no Web Audio: the game stays silent
  }

  effectsBus = ctx.createGain();
  alertsBus = ctx.createGain();
  effectsBus.connect(ctx.destination);
  alertsBus.connect(ctx.destination);
  applyVolumes();
  loadSamples(ctx);
}

function loadSamples(audioCtx: AudioContext) {
  const urls = new Set(Object.values(SOUNDS).flatMap((sound) => sound.src ?? []));
  for (const url of urls) {
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.arrayBuffer();
      })
      .then((data) => audioCtx.decodeAudioData(data))
      .then((buffer) => buffers.set(url, buffer))
      .catch((error) => console.warn(`Couldn't load ${url}`, error));
  }
}

function applyVolumes() {
  if (!ctx || !effectsBus || !alertsBus) return;

  const { muted, effectsVolume, alertsVolume } = get(soundSettings);
  const duck = effectsDucked ? FAST_FORWARD_DUCK : 1;
  // Squared, so the slider feels even to the ear
  effectsBus.gain.setTargetAtTime(muted ? 0 : effectsVolume ** 2 * duck, ctx.currentTime, VOLUME_RAMP_S);
  alertsBus.gain.setTargetAtTime(muted ? 0 : alertsVolume ** 2, ctx.currentTime, VOLUME_RAMP_S);
}

soundSettings.subscribe(applyVolumes);

// Browsers only let audio start after a user gesture; until then sounds are
// skipped rather than queued, so nothing bursts out later
export function unlockAudio() {
  if (!ctx) createContext();
  if (ctx?.state === "suspended") ctx.resume().catch(() => {});
}

// Fast-forward runs while the human is out, so its effects stay in the background
export function setEffectsDucked(ducked: boolean) {
  effectsDucked = ducked;
  applyVolumes();
}

export function playSound(name: SoundName, { gain = 1 }: { gain?: number } = {}) {
  const sound = SOUNDS[name];
  const bus = sound.channel === "alerts" ? alertsBus : effectsBus;
  if (!ctx || ctx.state !== "running" || !bus || get(soundSettings).muted) return;

  // Alerts always play; effects make way instead of piling up, and stay quiet
  // in a hidden tab so returning to it doesn't set off a burst
  if (sound.channel === "effects") {
    const now = performance.now();
    const sinceLast = now - (lastPlayed.get(name) ?? -Infinity);
    if (document.hidden || activeVoices >= MAX_VOICES || sinceLast < sound.cooldownMs) return;
    lastPlayed.set(name, now);
  }

  const voice = ctx.createGain();
  voice.gain.value = sound.gain * gain;
  const at = ctx.currentTime;
  let duration = 0;

  if (sound.src) {
    const loaded = sound.src
      .map((url) => buffers.get(url))
      .filter((buffer): buffer is AudioBuffer => !!buffer);
    if (loaded.length > 0) {
      const source = ctx.createBufferSource();
      source.buffer = loaded[Math.floor(Math.random() * loaded.length)];
      // A little variation, so repeats don't sound mechanical
      source.playbackRate.value = 0.95 + Math.random() * 0.1;
      voice.gain.value *= 0.85 + Math.random() * 0.15;
      source.connect(voice);
      source.start(at);
      duration = source.buffer.duration / source.playbackRate.value;
    } else if (!sound.synth) return; // still loading
  }
  if (sound.synth) duration = Math.max(duration, sound.synth(ctx, voice, at));

  voice.connect(bus);
  activeVoices++;
  setTimeout(() => {
    activeVoices--;
    voice.disconnect();
  }, duration * 1000 + 50);
}
