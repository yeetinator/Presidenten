<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import {
    connectionStatus,
    gameStore,
    logs,
    roundSummary,
    selectedCards,
    state as gameState,
  } from "./stores/gameStore";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";
  const defaultNumRounds = 10;

  const playerTypeOptions = [
    { value: 0, label: "Human" },
    { value: 1, label: "Random Bot" },
    { value: 2, label: "Baseline Bot" },
    { value: 3, label: "ISMCTS Bot" },
    { value: 4, label: "DMC Bot" },
  ] as const;

  let gameStarted = false;
  let showFinalResults = false;
  let totalPlayers = 4;
  let playerTypes: number[] = [1, 1, 1];
  let selectedIndices: number[] = [];

  $: requiredPlayerSlots = totalPlayers - 1;

  $: if (playerTypes.length !== requiredPlayerSlots) {
    playerTypes = Array.from(
      { length: requiredPlayerSlots },
      (_, index) => playerTypes[index] ?? 1,
    );
  }

  $: currentState = $gameState;
  $: lobbyLogs = $logs;
  $: roundSummaryData = $roundSummary;
  $: ownHand = currentState?.hand ?? [];
  $: cardsInPile = currentState?.cards_in_pile ?? [];
  $: selectedHandCards = $selectedCards;
  $: isSelectionLegal = (() => {
    if (!selectedMoveTuple || !currentState?.legal_moves) return false;
    return currentState.legal_moves.some(
      (move) =>
        move[0] === selectedMoveTuple[0] &&
        move[1] === selectedMoveTuple[1] &&
        move[2] === selectedMoveTuple[2],
    );
  })();
  $: opponentSeats = Object.entries(currentState?.player_roles ?? {})
    .map(([seatStr, role]) => {
      const seat = Number(seatStr);
      const relativeOffset = (seat - 0 + totalPlayers) % totalPlayers;
      return { seat, role, relativeOffset };
    })
    .filter((player) => player.seat !== 0)
    .sort((a, b) => a.relativeOffset - b.relativeOffset);
  $: leftOpponent = opponentSeats[0] ?? null;
  $: rightOpponent =
    opponentSeats.length > 1 ? opponentSeats[opponentSeats.length - 1] : null;
  $: topOpponents = opponentSeats.slice(1, -1);
  $: selectedMoveTuple = formatSelectedMove(selectedHandCards);
  $: selectedMoveLabel = selectedMoveTuple
    ? isSelectionLegal
      ? `Valid Move: ${selectedMoveTuple[1]}x ${displayCard(selectedMoveTuple[0])}`
      : "Illegal Move"
    : selectedHandCards.length > 0
      ? "Invalid selection"
      : "Click cards from your hand to queue them for play";
  $: roundSummaryEntries = roundSummaryData
    ? Object.entries(roundSummaryData.scores).sort(
        (left, right) => Number(left[0]) - Number(right[0]),
      )
    : [];
  $: if (currentState) {
    selectedIndices = [];
    gameStore.clearSelectedCards();
  }

  onMount(() => {
    gameStore.connect(websocketUrl);
  });

  onDestroy(() => {
    gameStore.disconnect();
  });

  function displayCard(card: number) {
    if (card === 11) return "J";
    if (card === 12) return "Q";
    if (card === 13) return "K";
    if (card === 14) return "A";
    if (card === 15) return "2";
    return String(card);
  }

  function displayRole(role: string | null) {
    return role ?? "Waiting";
  }

  function isValidSelectedMove(cards: number[]) {
    if (cards.length === 0) {
      return false;
    }

    const nonTwos = cards.filter((card) => card !== 15);
    if (nonTwos.length === 0) {
      return cards.length <= 4;
    }

    return nonTwos.every((card) => card === nonTwos[0]) && cards.length <= 4;
  }

  function formatSelectedMove(cards: number[]) {
    if (!isValidSelectedMove(cards)) {
      return null;
    }

    const twosUsed = cards.filter((card) => card === 15).length;
    const nonTwos = cards.filter((card) => card !== 15);

    if (nonTwos.length === 0) {
      return [15, cards.length, 0] as const;
    }

    const cardValue = nonTwos[0];
    return [cardValue, cards.length, twosUsed] as const;
  }

  async function handleStartGame() {
    const payload: {
      type: "START_GAME";
      num_players: number;
      num_rounds: number;
      player_types: number[];
    } = {
      type: "START_GAME",
      num_players: totalPlayers,
      num_rounds: defaultNumRounds,
      player_types: [0, ...playerTypes],
    };

    try {
      gameStore.clearLogs();
      gameStore.clearRoundSummary();
      gameStore.clearSelectedCards();
      await gameStore.startGame(payload);
      gameStarted = true;
      showFinalResults = false;
    } catch (error) {
      console.error(error);
    }
  }

  async function handlePlay() {
    try {
      await gameStore.playSelectedCards();
    } catch (error) {
      console.error(error);
    }
  }

  async function handlePass() {
    try {
      await gameStore.passTurn();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleNextRound() {
    try {
      await gameStore.nextRound();
    } catch (error) {
      console.error(error);
    }
  }

  function handleQuit() {
    showFinalResults = true;
    gameStore.quitGame();
  }

  function handleBackToLobby() {
    showFinalResults = false;
    gameStarted = false;
    selectedIndices = [];
    gameStore.clearSelectedCards();
    gameStore.clearRoundSummary();
    gameStore.clearLogs();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(index: number) {
    if (selectedIndices.includes(index)) {
      selectedIndices = selectedIndices.filter((i) => i !== index);
    } else {
      if (selectedIndices.length >= 4) return;
      selectedIndices = [...selectedIndices, index];
    }
    gameStore.selectedCards.set(selectedIndices.map((i) => ownHand[i]));
  }

  function handleManualClear() {
    selectedIndices = [];
    gameStore.clearSelectedCards();
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
            Player 0 is fixed to Human. Pick the remaining seats, then start the
            websocket game.
          </p>
        </div>

        <div class="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div class="rounded-2xl border border-white/10 bg-white/5 p-6">
            <div class="grid gap-4">
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
                        >
                          {option.label}
                        </option>
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
  <main
    class="min-h-screen bg-[radial-gradient(circle_at_center,#154d2a_0%,#0b2414_48%,#050b07_100%)] px-4 py-4 text-white md:px-6 md:py-6"
  >
    <section
      class="mx-auto grid min-h-[calc(100vh-2rem)] max-w-7xl grid-rows-[auto_auto_1fr_auto] gap-4 md:min-h-[calc(100vh-3rem)]"
    >
      <header
        class="rounded-3xl border border-white/10 bg-black/25 px-5 py-4 backdrop-blur-md md:px-6"
      >
        <div
          class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between"
        >
          <div>
            <p class="text-xs uppercase tracking-[0.35em] text-green-200/75">
              Digital Card Table
            </p>
            <h1 class="mt-1 text-2xl font-black md:text-4xl">Presidenten</h1>
          </div>
          <div class="flex flex-wrap gap-2 text-xs text-green-50/80">
            <span
              class="rounded-full border border-white/10 bg-white/5 px-3 py-1"
              >Status: {$connectionStatus}</span
            >
            <span
              class="rounded-full border border-white/10 bg-white/5 px-3 py-1"
              >Players: {totalPlayers}</span
            >
          </div>
        </div>
      </header>

      <section
        class="rounded-3xl border border-white/10 bg-black/25 px-5 py-4 backdrop-blur-md md:px-6"
      >
        <div class="flex items-center justify-between gap-3">
          <div>
            <p class="text-xs uppercase tracking-[0.35em] text-amber-200/75">
              Temporary Log Board
            </p>
            <h2 class="mt-1 text-lg font-semibold text-white">
              Lobby messages
            </h2>
          </div>
          <span class="text-xs text-slate-300"
            >Clears when a new game starts</span
          >
        </div>
        <div
          class="mt-4 max-h-40 overflow-y-auto rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-slate-200"
        >
          {#if lobbyLogs.length === 0}
            <p class="text-slate-400">No lobby messages yet.</p>
          {:else}
            <div class="space-y-2">
              {#each lobbyLogs as line}
                <div
                  class="rounded-xl border border-white/5 bg-white/5 px-3 py-2"
                >
                  {line}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </section>
      <div
        class="grid gap-4 lg:grid-cols-[1fr_1.3fr_1fr] lg:grid-rows-[auto_1fr]"
      >
        <div class="lg:col-span-3 lg:row-start-1">
          {#if topOpponents.length > 0}
            <div
              class="flex flex-wrap justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-md"
            >
              {#each topOpponents as opponent}
                <div
                  class="min-w-40 rounded-2xl border border-white/10 bg-black/25 px-4 py-3 text-center shadow-lg shadow-black/20"
                >
                  <div
                    class="text-[0.65rem] uppercase tracking-[0.3em] text-green-200/70"
                  >
                    Opponent {opponent.seat}
                  </div>
                  <div class="mt-2 text-lg font-bold">
                    {displayRole(opponent.role)}
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </div>

        <aside class="flex items-center justify-center lg:row-start-2">
          {#if leftOpponent}
            <div
              class="w-full max-w-sm rounded-3xl border border-white/10 bg-black/25 p-5 text-center shadow-2xl shadow-black/25 backdrop-blur-md"
            >
              <div class="text-xs uppercase tracking-[0.3em] text-green-200/70">
                Left Opponent
              </div>
              <div class="mt-2 text-2xl font-black">
                Player {leftOpponent.seat}
              </div>
              <div class="mt-1 text-sm text-green-50/75">
                {displayRole(leftOpponent.role)}
              </div>
            </div>
          {/if}
        </aside>

        <div class="flex items-center justify-center lg:row-start-2">
          <div
            class="w-full max-w-xl rounded-4xl border border-emerald-300/20 bg-[radial-gradient(circle_at_top,rgba(34,197,94,0.24),rgba(7,23,12,0.92)_70%)] p-6 shadow-[0_40px_100px_rgba(0,0,0,0.45)] backdrop-blur-md"
          >
            <div
              class="flex items-center justify-between text-sm text-emerald-100/75"
            >
              <span>Pile</span>
              <span>{cardsInPile.length} cards total</span>
            </div>
            <div
              class="mt-4 flex min-h-56 items-center justify-center rounded-3xl border border-emerald-300/15 bg-black/20 p-6"
            >
              {#if currentState?.last_move && currentState.last_move[0] !== 0}
                {@const [cardValue, count, twosUsed] = currentState.last_move}
                <div class="text-center">
                  <div
                    class="text-[0.65rem] uppercase tracking-[0.4em] text-emerald-200/70"
                  >
                    Current move to beat
                  </div>
                  <div class="mt-3 text-6xl font-black text-white">
                    {count}x {displayCard(cardValue)}
                    {#if twosUsed > 0}
                      <span class="text-xs text-emerald-400 block mt-1"
                        >(Includes {twosUsed}x 2)</span
                      >
                    {/if}
                  </div>
                </div>
              {:else}
                <div class="text-center text-emerald-50/70">
                  <div class="text-[0.65rem] uppercase tracking-[0.4em]">
                    Game Pile
                  </div>
                  <div class="mt-3 text-4xl font-black text-white/70">
                    Empty
                  </div>
                  <div class="mt-2 text-sm">
                    Play any valid combination to lead
                  </div>
                </div>
              {/if}
            </div>
          </div>
        </div>

        <aside class="flex items-center justify-center lg:row-start-2">
          {#if rightOpponent}
            <div
              class="w-full max-w-sm rounded-3xl border border-white/10 bg-black/25 p-5 text-center shadow-2xl shadow-black/25 backdrop-blur-md"
            >
              <div class="text-xs uppercase tracking-[0.3em] text-green-200/70">
                Right Opponent
              </div>
              <div class="mt-2 text-2xl font-black">
                Player {rightOpponent.seat}
              </div>
              <div class="mt-1 text-sm text-green-50/75">
                {displayRole(rightOpponent.role)}
              </div>
            </div>
          {/if}
        </aside>
      </div>

      <footer
        class="rounded-3xl border border-white/10 bg-black/25 p-4 backdrop-blur-md md:p-6"
      >
        <div class="grid gap-4">
          <div>
            <div class="flex items-center justify-between gap-3">
              <h2 class="text-lg font-semibold text-green-50">
                Selected Cards
              </h2>
              <div class="text-sm text-green-100/70">{selectedMoveLabel}</div>
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
              {#if selectedHandCards.length === 0}
                <span
                  class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm text-green-50/70"
                  >No cards selected</span
                >
              {:else}
                {#each selectedHandCards as card, index}
                  <span
                    class="rounded-full border border-emerald-300/20 bg-emerald-400/10 px-3 py-1 text-sm text-emerald-100"
                  >
                    {displayCard(card)}
                  </span>
                {/each}
              {/if}
            </div>
          </div>

          <div>
            <div class="flex items-center justify-between gap-3">
              <h2 class="text-lg font-semibold text-green-50">Your Hand</h2>
              {#if selectedHandCards.length > 0}
                <button
                  type="button"
                  on:click={handleManualClear}
                  class="text-xs text-red-400 hover:text-red-300 font-medium underline"
                >
                  Clear Selection
                </button>
              {/if}
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
              {#if ownHand.length === 0}
                <span
                  class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm text-green-50/70"
                  >No hand data yet</span
                >
              {:else}
                {#each ownHand as card, index}
                  <button
                    class="min-w-14 rounded-2xl border px-4 py-3 text-lg font-black shadow-lg transition hover:-translate-y-0.5 active:translate-y-0
                    {selectedIndices.includes(index)
                      ? 'bg-emerald-400 text-emerald-950 border-emerald-300 scale-105 ring-2 ring-emerald-400/50'
                      : 'border-white/10 bg-white text-slate-950 hover:bg-slate-100'}"
                    type="button"
                    on:click={() => handleToggleCard(index)}
                  >
                    {displayCard(card)}
                  </button>
                {/each}
              {/if}
            </div>
          </div>

          <div
            class="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 p-4"
          >
            <div class="text-sm text-green-50/75">{selectedMoveLabel}</div>
            <div class="flex flex-wrap gap-3">
              <button
                class="rounded-xl border border-emerald-300/30 bg-emerald-400 px-5 py-3 font-semibold text-emerald-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-40"
                type="button"
                disabled={!isSelectionLegal}
                on:click={handlePlay}
              >
                Play
              </button>
              <button
                class="rounded-xl border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
                type="button"
                on:click={handlePass}
              >
                Pass
              </button>
            </div>
          </div>
        </div>
      </footer>

      {#if showFinalResults}
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
                <p
                  class="text-xs uppercase tracking-[0.35em] text-emerald-300/80"
                >
                  Game ended
                </p>
                <h2 class="mt-1 text-2xl font-black text-white">
                  Final Results
                </h2>
              </div>
              <button
                class="rounded-xl border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
                type="button"
                on:click={handleBackToLobby}
              >
                Back to Lobby
              </button>
            </div>

            {#if roundSummaryData}
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
                {#each roundSummaryEntries as [seat, [points, wins]]}
                  <div
                    class="grid grid-cols-[0.7fr_1fr_1fr] gap-px bg-white/10 text-sm text-slate-100"
                  >
                    <div class="bg-black/30 px-4 py-3 font-semibold">
                      Player {seat}
                    </div>
                    <div class="bg-black/30 px-4 py-3">{points}</div>
                    <div class="bg-black/30 px-4 py-3">{wins}</div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {:else if roundSummaryData}
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
                <p
                  class="text-xs uppercase tracking-[0.35em] text-amber-200/75"
                >
                  Round complete
                </p>
                <h2 class="mt-1 text-2xl font-black text-white">
                  Play the next round or quit
                </h2>
              </div>
              <div class="flex gap-3">
                <button
                  class="rounded-xl bg-emerald-400 px-5 py-3 font-semibold text-emerald-950 transition hover:bg-emerald-300"
                  type="button"
                  on:click={handleNextRound}
                >
                  Play Next Round
                </button>
                <button
                  class="rounded-xl border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
                  type="button"
                  on:click={handleQuit}
                >
                  Quit
                </button>
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
              {#each roundSummaryEntries as [seat, [points, wins]]}
                <div
                  class="grid grid-cols-[0.7fr_1fr_1fr] gap-px bg-white/10 text-sm text-slate-100"
                >
                  <div class="bg-black/30 px-4 py-3 font-semibold">
                    Player {seat}
                  </div>
                  <div class="bg-black/30 px-4 py-3">{points}</div>
                  <div class="bg-black/30 px-4 py-3">{wins}</div>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {/if}
    </section>
  </main>
{/if}
