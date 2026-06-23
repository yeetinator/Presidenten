<script lang="ts">
  import Card from "../../assets/Card.svelte";

  type OpponentView = {
    seat: number;
    role: string | null;
    handCount: number;
    label: string;
  };

  export let opponent: OpponentView;
  export let className = "";
</script>

<article class={`flex flex-col ${className}`}>
  <div class="flex items-center justify-center gap-2">
    <img
      src="/bot.svg"
      alt="Bot profile icon"
      class="h-7 w-7 rounded-lg border border-white/10 bg-white/10 p-1.5"
    />
    <div class="min-w-0">
      <div class="truncate text-sm font-black text-white">{opponent.label}</div>
    </div>
  </div>
  <div class="mt-6 flex h-10 w-full items-end justify-center overflow-visible">
    {#each Array.from({ length: opponent.handCount }) as _, cardIndex}
      <div
        class="relative"
        style={`margin-left: ${cardIndex === 0 ? 0 : -5.4}rem; z-index: ${cardIndex};`}
      >
        <Card
          suitCard="3C"
          isFaceUp={false}
          disabled={true}
          className="shrink-0 scale-[0.4] origin-bottom"
        />
      </div>
    {/each}
  </div>
</article>
