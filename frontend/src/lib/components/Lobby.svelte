<script lang="ts">
  type StartGameConfig = {
    numPlayers: number;
    numRounds: number;
    playerTypes: number[];
  };

  export let onStartGame:
    | ((config: StartGameConfig) => void | Promise<void>)
    | undefined = undefined;

  const playerTypeOptions = [
    { value: 0, label: "Human" },
    { value: 1, label: "Random" },
    { value: 2, label: "Baseline" },
    { value: 4, label: "DMC" },
  ] as const;

  const defaultNumRounds = 10;

  let totalPlayers = 4;
  let playerTypes: number[] = [1, 1, 1];

  $: requiredPlayerSlots = totalPlayers - 1;

  $: if (playerTypes.length !== requiredPlayerSlots) {
    playerTypes = Array.from(
      { length: requiredPlayerSlots },
      (_, index) => playerTypes[index] ?? 1,
    );
  }

  async function handleStartGame() {
    try {
      await onStartGame?.({
        numPlayers: totalPlayers,
        numRounds: defaultNumRounds,
        playerTypes: [0, ...playerTypes],
      });
    } catch (error) {
      console.error(error);
    }
  }

  function setPlayerType(slotIndex: number, value: number) {
    playerTypes = playerTypes.map((currentValue, currentIndex) =>
      currentIndex === slotIndex ? value : currentValue,
    );
  }
</script>

<main
  class="min-h-screen bg-[radial-gradient(circle_at_top,#1f7a3d_0%,#0f3d20_48%,#07170c_100%)] px-6 py-10 text-white"
>
  <div
    class="mx-auto flex min-h-[calc(100vh-5rem)] max-w-5xl items-center justify-center"
  >
    <section
      class="w-full rounded-3xl border border-white/10 bg-black/25 p-8 shadow-2xl shadow-black/35 backdrop-blur-md md:p-10"
    >
      <div class="mb-8 space-y-3">
        <p class="text-sm uppercase tracking-[0.35em] text-green-200/80">
          President Lobby
        </p>
        <h1 class="text-4xl font-black tracking-tight md:text-6xl">
          Configure the match
        </h1>
        <p class="max-w-2xl text-sm text-green-50/80 md:text-base">
          Player 0 is fixed to Human. Pick the remaining seats, then start the
          websocket game.
        </p>
      </div>

      <div class="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
          <div class="grid gap-4">
            <label class="grid gap-2">
              <span class="text-sm font-semibold text-green-100">
                Total Players
              </span>
              <select
                class="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-white outline-none transition focus:border-green-300 focus:ring-2 focus:ring-green-300/30"
                bind:value={totalPlayers}
              >
                {#each Array(4) as _, index}
                  <option
                    value={index + 4}
                    class="bg-[#0f3d20] text-white select-none"
                  >
                    {index + 4} Players
                  </option>
                {/each}
              </select>
            </label>

            <button
              class="mt-2 rounded-xl bg-green-400 px-5 py-3 font-semibold text-green-950 transition hover:bg-green-300 active:scale-[0.99]"
              type="button"
              on:click={handleStartGame}
            >
              Start Game
            </button>
          </div>
        </div>

        <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
          <div class="mb-4 flex items-center justify-between">
            <h2 class="text-lg font-semibold text-green-50">Player setup</h2>
            <span class="text-sm text-green-100/70">Player 0: Human</span>
          </div>

          <div class="grid gap-4">
            {#each Array(requiredPlayerSlots) as _, index}
              <label
                class="grid gap-2 rounded-xl border border-white/10 bg-black/20 p-4"
              >
                <span class="text-sm font-semibold text-green-100">
                  Player {index + 1}
                </span>
                <div class="grid grid-cols-3 gap-2">
                  {#each playerTypeOptions.filter((option) => option.value !== 0) as option}
                    <button
                      class={`rounded-lg border px-3 py-2 text-sm font-semibold transition active:scale-[0.99] ${playerTypes[index] === option.value ? "border-green-300 bg-green-400 text-green-950" : "border-white/10 bg-black/30 text-white hover:border-green-300/60 hover:bg-black/40"}`}
                      type="button"
                      aria-pressed={playerTypes[index] === option.value}
                      on:click={() => setPlayerType(index, option.value)}
                    >
                      {option.label}
                    </button>
                  {/each}
                </div>
              </label>
            {/each}
          </div>
        </div>
      </div>
    </section>
  </div>
</main>
