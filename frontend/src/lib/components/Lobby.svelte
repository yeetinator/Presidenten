<script lang="ts">
  import Rules from "../../assets/Rules.svelte";
  import { currTheme, setTheme } from "../../stores/gameStore";
  import { TABLE_THEMES } from "../themes";

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
    playerTypes = playerTypes.map((currValue, currIndex) =>
      currIndex === slotIndex ? value : currValue,
    );
  }
</script>

<main
  class="min-h-screen px-6 py-10 text-white transition-colors duration-500"
  style={`background: ${$currTheme.mainBg};`}
>
  <div
    class="mx-auto flex min-h-[calc(100vh-5rem)] max-w-5xl items-center justify-center"
  >
    <section
      class="w-full rounded-3xl border border-white/10 bg-black/25 p-8 shadow-2xl shadow-black/35 backdrop-blur-md md:p-10"
    >
      <Rules className="right-4" side="right" />
      <div class="mb-8 space-y-3">
        <p class="text-sm uppercase tracking-[0.35em] text-white/70">
          President Lobby
        </p>
        <h1 class="text-4xl font-black tracking-tight md:text-6xl text-white">
          Configure the match
        </h1>
        <p class="max-w-2xl text-sm text-slate-200/90 md:text-base">
          Player 0 is fixed to Human. Pick the remaining seats, then start the
          websocket game.
        </p>
      </div>

      <div class="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
          <div class="grid gap-4">
            <label class="grid gap-2">
              <span class="text-sm font-semibold text-white/90">
                Total Players
              </span>
              <select
                class="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-white outline-none transition focus:border-white/40 focus:ring-2 focus:ring-white/20"
                bind:value={totalPlayers}
              >
                {#each Array(4) as _, index}
                  <option
                    value={index + 4}
                    class="bg-slate-900 text-white select-none"
                  >
                    {index + 4} Players
                  </option>
                {/each}
              </select>
            </label>

            <button
              class="mt-2 rounded-xl px-5 py-3 font-semibold transition hover:opacity-90 active:scale-[0.99]"
              style={`background-color: ${$currTheme.btnBg}; color: ${$currTheme.btnText};`}
              type="button"
              on:click={handleStartGame}
            >
              Start Game
            </button>

            <div class="mt-2 grid gap-2.5 border-t border-white/10 pt-4">
              <span
                class="text-xs font-semibold uppercase tracking-wider text-white/70"
              >
                Table Felt Theme
              </span>
              <div class="grid grid-cols-2 gap-2">
                {#each Object.values(TABLE_THEMES) as theme}
                  <button
                    type="button"
                    class={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-xs font-semibold transition active:scale-[0.98] ${
                      $currTheme.id === theme.id
                        ? "bg-white/15 text-white shadow-sm"
                        : "border-white/10 bg-black/30 text-white/70 hover:border-white/20 hover:bg-black/50"
                    }`}
                    on:click={() => setTheme(theme.id)}
                  >
                    <span
                      class="h-4 w-4 shrink-0 rounded-full border border-white/30 shadow-inner"
                      style={`background-color: ${theme.swatchColor};`}
                    ></span>
                    <span class="truncate">{theme.name}</span>
                  </button>
                {/each}
              </div>
            </div>
          </div>
        </div>

        <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
          <div class="mb-4 flex items-center justify-between">
            <h2 class="text-lg font-semibold text-white">Player setup</h2>
            <span class="text-sm text-slate-300">Player 0: Human</span>
          </div>

          <div class="grid gap-4">
            {#each Array(requiredPlayerSlots) as _, index}
              <div
                class="grid gap-2 rounded-xl border border-white/10 bg-black/20 p-4"
              >
                <span class="text-sm font-semibold text-white/90">
                  Player {index + 1}
                </span>
                <div class="grid grid-cols-3 gap-2">
                  {#each playerTypeOptions.filter((option) => option.value !== 0) as option}
                    <button
                      class="rounded-lg border px-3 py-2 text-sm font-semibold transition active:scale-[0.99]"
                      style={playerTypes[index] === option.value
                        ? `background-color: ${$currTheme.btnBg}; color: ${$currTheme.btnText}; border-color: ${$currTheme.accentBorder};`
                        : "border-color: rgba(255, 255, 255, 0.1); background-color: rgba(0, 0, 0, 0.3); color: #ffffff;"}
                      type="button"
                      aria-pressed={playerTypes[index] === option.value}
                      on:click={() => setPlayerType(index, option.value)}
                    >
                      {option.label}
                    </button>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </section>
  </div>
</main>
