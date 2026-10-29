<script lang="ts">
  import { fly } from "svelte/transition";
  import { playSound, soundSettings, type SoundName } from "../audio";

  export let className = "";
  export let side: "left" | "right" = "right";
  // Previews would talk over the game, so they are only offered in menus
  export let previews = false;

  type VolumeField = "effectsVolume" | "alertsVolume";

  const volumeSliders: { field: VolumeField; label: string; preview: SoundName }[] = [
    { field: "effectsVolume", label: "Effects", preview: "cardPlay" },
    { field: "alertsVolume", label: "Alerts", preview: "jumpIn" },
  ];

  let isOpen = false;
  let root: HTMLDivElement;

  function toggleMuted() {
    soundSettings.update((settings) => ({ ...settings, muted: !settings.muted }));
  }

  function setVolume(field: VolumeField, event: Event) {
    const volume = Number((event.currentTarget as HTMLInputElement).value) / 100;
    soundSettings.update((settings) => ({ ...settings, [field]: volume }));
  }

  function preview(name: SoundName) {
    if (previews) playSound(name);
  }

  function handleWindowPointerDown(event: PointerEvent) {
    if (isOpen && !root.contains(event.target as Node)) isOpen = false;
  }

  function handleWindowKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") isOpen = false;
  }
</script>

<svelte:window
  on:pointerdown={handleWindowPointerDown}
  on:keydown={handleWindowKeyDown}
/>

<div bind:this={root} class={`fixed top-4 ${className} z-30`}>
  <button
    type="button"
    class="flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-white/10
           text-white shadow-lg shadow-black/20 backdrop-blur-md transition-all duration-200
           hover:border-white/30 hover:bg-white/20 active:scale-95"
    aria-label="Sound settings"
    aria-expanded={isOpen}
    on:click={() => (isOpen = !isOpen)}
  >
    <svg
      viewBox="0 0 24 24"
      class="h-5 w-5"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M11 5 6 9H3v6h3l5 4z" fill="currentColor" />
      {#if $soundSettings.muted}
        <path d="m16 9 5 6m0-6-5 6" />
      {:else}
        <path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />
      {/if}
    </svg>
  </button>

  {#if isOpen}
    <div
      in:fly={{ y: -8, duration: 200 }}
      out:fly={{ y: -6, duration: 150 }}
      class={`absolute ${side === "left" ? "left-0" : "right-0"} top-12 grid w-64 gap-4 rounded-2xl border
             border-white/10 bg-slate-950/95 p-5 text-sm text-white shadow-2xl shadow-black/50 backdrop-blur-xl`}
    >
      <div class="flex items-center justify-between">
        <h3 class="font-black uppercase tracking-wider text-amber-400">Sound</h3>
        <button
          type="button"
          class={`rounded-lg border px-3 py-1 text-xs font-semibold transition active:scale-[0.98] ${
            $soundSettings.muted
              ? "border-red-300/30 bg-red-500/80 text-white"
              : "border-white/10 bg-white/10 text-white/90 hover:bg-white/15"
          }`}
          aria-pressed={$soundSettings.muted}
          on:click={toggleMuted}
        >
          {$soundSettings.muted ? "Muted" : "Mute"}
        </button>
      </div>

      {#each volumeSliders as slider}
        <label class={`grid gap-1.5 ${$soundSettings.muted ? "opacity-50" : ""}`}>
          <span class="flex justify-between text-xs font-semibold text-white/80">
            {slider.label}
            <span class="text-white/50">{Math.round($soundSettings[slider.field] * 100)}%</span>
          </span>
          <input
            type="range"
            min="0"
            max="100"
            value={Math.round($soundSettings[slider.field] * 100)}
            class="w-full accent-amber-400"
            on:input={(event) => setVolume(slider.field, event)}
            on:change={() => preview(slider.preview)}
          />
        </label>
      {/each}

      {#if previews}
        <button
          type="button"
          class="rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-xs font-semibold text-white/90
                 transition hover:bg-white/15 active:scale-[0.98]"
          on:click={() => playSound("jumpIn")}
        >
          Test jump-in alert
        </button>
      {/if}
    </div>
  {/if}
</div>
