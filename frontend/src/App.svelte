<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";

  const playerTypeOptions = [
    { value: 0, label: "Human" },
    { value: 1, label: "Random Bot" },
    { value: 2, label: "Baseline Bot" },
    { value: 3, label: "ISMCTS Bot" },
    { value: 4, label: "DMC Bot" },
  ] as const;

  let gameStarted = false;
  let totalPlayers = 4;
  let numRounds = 10;
  let playerTypes: number[] = [1, 1, 1];
  let socket: WebSocket | null = null;

  $: requiredPlayerSlots = totalPlayers - 1;

  $: if (playerTypes.length !== requiredPlayerSlots) {
    playerTypes = Array.from(
      { length: requiredPlayerSlots },
      (_, index) => playerTypes[index] ?? 1,
    );
  }

  onMount(() => {
    socket = new WebSocket(websocketUrl);
  });

  onDestroy(() => {
    socket?.close();
  });

  function sendStartGameMessage(payload: Record<string, unknown>) {
    const message = JSON.stringify(payload);

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(message);
      return Promise.resolve();
    }

    if (!socket || socket.readyState === WebSocket.CLOSED) {
      socket = new WebSocket(websocketUrl);
    }

    return new Promise<void>((resolve, reject) => {
      if (!socket) {
        reject(new Error("WebSocket is not available."));
        return;
      }

      const handleOpen = () => {
        socket?.send(message);
        cleanup();
        resolve();
      };

      const handleError = () => {
        cleanup();
        reject(new Error("Unable to open the game websocket."));
      };

      const cleanup = () => {
        socket?.removeEventListener("open", handleOpen);
        socket?.removeEventListener("error", handleError);
      };

      socket.addEventListener("open", handleOpen);
      socket.addEventListener("error", handleError);
    });
  }

  async function handleStartGame() {
    const payload = {
      type: "START_GAME",
      num_players: totalPlayers,
      num_rounds: numRounds,
      player_types: [0, ...playerTypes],
    };

    try {
      await sendStartGameMessage(payload);
      gameStarted = true;
    } catch (error) {
      console.error(error);
    }
  }
</script>

{#if !gameStarted}
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
            Presidenten Lobby
          </p>
          <h1 class="text-4xl font-black tracking-tight md:text-6xl">
            Configure the match
          </h1>
          <p class="max-w-2xl text-sm text-green-50/80 md:text-base">
            Player 0 is fixed to Human. Pick the remaining seats, set the round
            count, then start the websocket game.
          </p>
        </div>

        <div class="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
            <div class="grid gap-4">
              <label class="grid gap-2">
                <span class="text-sm font-semibold text-green-100">Rounds</span>
                <input
                  class="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-white outline-none transition focus:border-green-300 focus:ring-2 focus:ring-green-300/30"
                  type="number"
                  min="1"
                  bind:value={numRounds}
                />
              </label>

              <label class="grid gap-2">
                <span class="text-sm font-semibold text-green-100"
                  >Total Players</span
                >
                <select
                  class="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-white outline-none transition focus:border-green-300 focus:ring-2 focus:ring-green-300/30"
                  bind:value={totalPlayers}
                >
                  {#each Array(4) as _, index}
                    <option
                      value={index + 4}
                      class="bg-[#0f3d20] text-white select-none"
                      >{index + 4} Players</option
                    >
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
                  <span class="text-sm font-semibold text-green-100"
                    >Player {index + 1}</span
                  >
                  <select
                    class="rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-white outline-none transition focus:border-green-300 focus:ring-2 focus:ring-green-300/30"
                    bind:value={playerTypes[index]}
                  >
                    {#each playerTypeOptions as option}
                      {#if option.value !== 0}
                        <option
                          value={option.value}
                          class="bg-[#0f3d20] text-white select-none"
                          >{option.label}</option
                        >
                      {/if}
                    {/each}
                  </select>
                </label>
              {/each}
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>
{:else}
  <main class="min-h-screen bg-slate-950 px-6 py-10 text-white">
    <div
      class="mx-auto flex min-h-[calc(100vh-5rem)] max-w-5xl items-center justify-center"
    >
      <section
        class="w-full rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl shadow-black/35 backdrop-blur-md md:p-10"
      >
        <p class="text-sm uppercase tracking-[0.35em] text-emerald-300/80">
          Game started
        </p>
        <h1 class="mt-3 text-4xl font-black tracking-tight md:text-6xl">
          Table view
        </h1>
        <p class="mt-4 max-w-2xl text-sm text-slate-300 md:text-base">
          The START_GAME payload has been sent. Replace this screen with the
          live game UI.
        </p>

        <div
          class="mt-8 grid gap-4 rounded-2xl border border-white/10 bg-black/20 p-6 text-sm text-slate-200 md:grid-cols-3"
        >
          <div>
            <div class="text-slate-400">Rounds</div>
            <div class="mt-1 text-2xl font-bold text-white">{numRounds}</div>
          </div>
          <div>
            <div class="text-slate-400">Players</div>
            <div class="mt-1 text-2xl font-bold text-white">{totalPlayers}</div>
          </div>
          <div>
            <div class="text-slate-400">Human seat</div>
            <div class="mt-1 text-2xl font-bold text-white">0</div>
          </div>
        </div>
      </section>
    </div>
  </main>
{/if}
