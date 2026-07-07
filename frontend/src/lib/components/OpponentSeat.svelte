<script lang="ts">
  import Card from "../../assets/Card.svelte";
  import { send, receive } from "../../lib/transitions";
  import {
    fastForwardMode,
    startAnimation,
    endAnimation,
  } from "../../stores/gameStore";

  type OpponentView = {
    seat: number;
    role: string | null;
    suitedHand: string[];
    label: string;
    is_turn: boolean;
    has_passed: boolean;
  };

  export let opponent: OpponentView;
  export let className = "";
  export let revealBotCards = false;
</script>

<article
  class={`flex flex-col gap-1 transition-all duration-300 ${className} ${opponent.has_passed ? "opacity-40 grayscale pointer-events-none" : ""}`}
  style={`transition-duration: ${$fastForwardMode ? "100ms" : "200ms"};`}
>
  <div class="flex items-center justify-center gap-2">
    <img
      src="/bot.svg"
      alt="Bot profile icon"
      class="h-7 w-7 rounded-lg border border-white/10 bg-white/10 p-1.5"
    />
    <div class="min-w-0">
      <div
        class={`truncate text-sm font-black ${opponent.is_turn ? "text-yellow-400" : "text-white"}`}
      >
        {opponent.label}
      </div>
    </div>
  </div>
  <div class="mt-6 flex h-10 w-full items-end justify-center overflow-visible">
    {#each opponent.suitedHand as suitCard, cardIndex (suitCard)}
      <div
        in:receive={{ key: suitCard }}
        out:send={{ key: suitCard }}
        on:introstart={startAnimation}
        on:introend={endAnimation}
        on:outrostart={startAnimation}
        on:outroend={endAnimation}
        class="relative transition-all duration-300"
        style={`transition-duration: ${$fastForwardMode ? "100ms" : "200ms"}; margin-left: ${cardIndex === 0 ? 0 : -5.4}rem; z-index: ${cardIndex};`}
      >
        <Card
          {suitCard}
          isFaceUp={revealBotCards}
          disabled={true}
          className="shrink-0 scale-[0.4] origin-bottom"
        />
      </div>
    {/each}
  </div>
</article>
