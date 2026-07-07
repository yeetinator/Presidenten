<script lang="ts">
  import { fly } from "svelte/transition";

  export let className = "";
  export let side: "left" | "right" = "right";

  let isHovered = false;
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class={`fixed top-4 ${className} z-30 flex h-9 w-9 items-center justify-center rounded-full
         border border-white/20 bg-white/10 shadow-lg shadow-black/20
         backdrop-blur-md transition-all duration-200
         hover:border-white/30 hover:bg-white/20 active:scale-95`}
  on:mouseenter={() => (isHovered = true)}
  on:mouseleave={() => (isHovered = false)}
>
  <img
    src="/info.svg"
    alt="Info button"
    class="h-full w-full opacity-90 invert brightness-0"
  />

  {#if isHovered}
    <div
      in:fly={{ x: -15, duration: 200 }}
      out:fly={{ x: -10, duration: 150 }}
      class={`absolute ${side === "left" ? "left-12" : "right-12"} top-0 w-96 rounded-2xl border border-white/10
             bg-slate-950/95 p-5 text-sm shadow-2xl shadow-black/50 backdrop-blur-xl`}
    >
      <div class="mb-3 border-b border-white/10 pb-2">
        <h3 class="font-black uppercase tracking-wider text-amber-400">
          President Rules
        </h3>
        <p class="text-[0.7rem] text-slate-400">Quick game guide</p>
      </div>
      <div class="space-y-3 text-xs leading-relaxed text-slate-200">
        <p>
          <strong class="text-white">Objective:</strong> Shed your hand as fast as
          possible to earn the highest social rank for the next round.
        </p>

        <ul class="list-disc space-y-1.5 pl-4 text-slate-300">
          <li>
            <span class="text-white font-medium">Setup:</span> 3 of Clubs leads the
            first round. The Scum leads all subsequent rounds.
          </li>
          <li>
            <span class="text-white font-medium">Ranks:</span> Cards range from 3
            (lowest) up to Ace. 2s are the ultimate trumps.
          </li>
          <li>
            <span class="text-white font-medium">Turn Order:</span> Play moves clockwise.
            You must match the combination type (single, pair, etc.) but play a higher
            value, or pass.
          </li>
          <li>
            <span class="text-white font-medium">Wild Cards (2s):</span> Beats any
            single card solo, or acts as a wild matching card to fill out sets (e.g.,
            3 + 2 = pair of 3s).
          </li>
          <li>
            <span class="text-white font-medium">Clearing:</span> Resets on consecutive
            passes or an instant Jump-In. Last player to act opens the new pile.
            Solo 2s clear the pile instantly unless a Jump-In occurs.
          </li>
          <li>
            <span class="text-white font-medium">Jump-In:</span> Play immediately
            out of turn if your cards can complete a four-of-a-kind set instantly.
          </li>
        </ul>
      </div>
    </div>
  {/if}
</div>
