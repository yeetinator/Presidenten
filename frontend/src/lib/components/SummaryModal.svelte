<script lang="ts">
  type ScoreRow = [string, [number, number]];

  export let eyebrow = "";
  export let eyebrowClass = "text-emerald-300/80";
  export let title = "";
  export let scores: ScoreRow[] = [];

  $: sortedScores = [...scores].sort(
    (left, right) => Number(right[1][0]) - Number(left[1][0]),
  );
</script>

<div
  class="fixed inset-0 z-20 flex items-center justify-center bg-black/70 px-4 py-8 backdrop-blur-sm"
>
  <div
    class="w-full max-w-3xl rounded-3xl border border-white/10 bg-slate-950 p-6 shadow-2xl shadow-black/50 md:p-8"
  >
    <div
      class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between"
    >
      <div>
        <p class={`text-xs uppercase tracking-[0.35em] ${eyebrowClass}`}>
          {eyebrow}
        </p>
        <h2 class="mt-1 text-2xl font-black text-white">{title}</h2>
      </div>
      <div class="flex gap-3">
        <slot name="actions" />
      </div>
    </div>

    <div
      class="mt-6 overflow-hidden rounded-2xl border border-white/10 bg-black/20"
    >
      <div
        class="grid grid-cols-[0.7fr_1fr_1fr] gap-px bg-white/10 text-sm font-semibold text-green-100"
      >
        <div class="bg-black/40 px-4 py-3">Player</div>
        <div class="bg-black/40 px-4 py-3">Total Points</div>
        <div class="bg-black/40 px-4 py-3">Wins</div>
      </div>
      {#each sortedScores as [seat, [points, wins]]}
        <div
          class="grid grid-cols-[0.7fr_1fr_1fr] gap-px bg-white/10 text-sm text-slate-100"
        >
          <div class="bg-black/30 px-4 py-3 font-semibold">Player {seat}</div>
          <div class="bg-black/30 px-4 py-3">{points}</div>
          <div class="bg-black/30 px-4 py-3">{wins}</div>
        </div>
      {/each}
    </div>
  </div>
</div>
