<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { cubicIn } from "svelte/easing";
  import { Tween } from "svelte/motion";
  import Card from "./lib/Card.svelte";
  import {
    connectionStatus,
    gameStore,
    exchangePrompt,
    jumpInPrompt,
    roundSummary,
    selectedCards,
    state as gameState,
    type VisualCard,
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
  let visualHand: VisualCard[] = [];
  let lastHandChecksum = "";

  const jumpInShake = new Tween(0, { duration: 1500, easing: cubicIn });

  $: requiredPlayerSlots = totalPlayers - 1;

  $: if (playerTypes.length !== requiredPlayerSlots) {
    playerTypes = Array.from(
      { length: requiredPlayerSlots },
      (_, index) => playerTypes[index] ?? 1,
    );
  }

  $: currentState = $gameState;
  $: exchangePromptData = $exchangePrompt;
  $: jumpInPromptData = $jumpInPrompt;
  $: roundSummaryData = $roundSummary;
  $: ownHand = currentState?.hand ?? [];
  $: cardsInPile = currentState?.cards_in_pile ?? [];
  $: selectedHandCards = $selectedCards;
  $: exchangeRequiredCards = exchangePromptData?.requiredCards ?? 0;
  $: exchangeCanChoose = exchangePromptData?.canChoose ?? false;
  $: isExchangeVisible = !!exchangePromptData;
  $: jumpInVisible = !!jumpInPromptData && !!currentState?.is_finish_prompt;
  $: isSelectionLegal = (() => {
    if (!selectedMoveTuple || !currentState?.legal_moves || isExchangeVisible)
      return false;
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
  $: opponentViews = opponentSeats.map((opponent, index) => ({
    ...opponent,
    handCount: currentState?.opp_hand_counts?.[opponent.seat] ?? 0,
    position:
      index === 0
        ? "left"
        : index === opponentSeats.length - 1
          ? "right"
          : "top",
  }));
  $: leftOpponentView =
    opponentViews.find((opponent) => opponent.position === "left") ?? null;
  $: rightOpponentView =
    opponentViews.find((opponent) => opponent.position === "right") ?? null;
  $: topOpponentViews = opponentViews.filter(
    (opponent) => opponent.position === "top",
  );
  $: selectedMoveTuple = formatSelectedMove(selectedHandCards);
  $: selectedMoveLabel = selectedMoveTuple
    ? isExchangeVisible
      ? `Selected ${selectedHandCards.length}/${exchangeRequiredCards} cards`
      : isSelectionLegal
        ? `Valid Move: ${selectedMoveTuple[1]}x ${displayCard(selectedMoveTuple[0])}`
        : "Illegal Move"
    : selectedHandCards.length > 0
      ? isExchangeVisible
        ? `Selected ${selectedHandCards.length}/${exchangeRequiredCards} cards`
        : "Invalid selection"
      : isExchangeVisible
        ? exchangeRequiredCards === 0
          ? "Citizen's are exempt from exchanging"
          : exchangeCanChoose
            ? `Select ${exchangeRequiredCards} cards to give away`
            : `Highest ${exchangeRequiredCards} cards pre-selected`
        : "Click cards from your hand to queue them for play";
  $: roundSummaryEntries = roundSummaryData
    ? Object.entries(roundSummaryData.scores).sort(
        (left, right) => Number(left[0]) - Number(right[0]),
      )
    : [];
  $: if (jumpInVisible) {
    jumpInShake.target = 1;
  } else {
    jumpInShake.target = 0;
  }

  let lastStateRef: typeof currentState = null;
  $: if (currentState && !isExchangeVisible) {
    if (currentState !== lastStateRef) {
      lastStateRef = currentState;
      selectedIndices = [];
      gameStore.clearSelectedCards();
    }
  }

  let lastExchangeActive = false;
  $: if (isExchangeVisible !== lastExchangeActive) {
    lastExchangeActive = isExchangeVisible;
    if (isExchangeVisible) {
      if (exchangeCanChoose) {
        selectedIndices = [];
        gameStore.clearSelectedCards();
      } else if (exchangeRequiredCards > 0 && currentState) {
        selectedIndices = Array.from(
          { length: exchangeRequiredCards },
          (_, index) =>
            currentState.hand.length - exchangeRequiredCards + index,
        );
      }
    }
  }

  $: if (ownHand.length > 0) {
    const checksum = ownHand.join(",");
    if (checksum !== lastHandChecksum) {
      syncVisualHand(ownHand);
      lastHandChecksum = checksum;
    }
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

  function displayBotType(role: string | null) {
    if (!role) return "Waiting";
    return role.replace(/\s*Bot$/i, "");
  }

  function getFanSpacingClass(
    handCount: number,
    orientation: "horizontal" | "vertical",
  ) {
    if (handCount <= 3) {
      return orientation === "horizontal" ? "-space-x-3" : "-space-y-3";
    }

    if (handCount <= 6) {
      return orientation === "horizontal" ? "-space-x-4" : "-space-y-4";
    }

    if (handCount <= 9) {
      return orientation === "horizontal" ? "-space-x-5" : "-space-y-5";
    }

    return orientation === "horizontal" ? "-space-x-6" : "-space-y-6";
  }

  function getFanDirectionClass(orientation: "horizontal" | "vertical") {
    return orientation === "horizontal"
      ? "flex-row items-end"
      : "flex-col items-center";
  }

  function getOpponentLabel(role: string | null) {
    return displayBotType(role);
  }

  function getHorizontalOverlapStyle(index: number) {
    return index === 0 ? "" : "margin-left: -4rem;";
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
      gameStore.clearExchangePrompt();
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

  async function handleJumpIn() {
    try {
      await gameStore.playJumpInPrompt();
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

  async function handleConfirmExchange() {
    if (!currentState || !exchangePromptData) return;

    const cards = selectedHandCards;
    if (cards.length !== exchangeRequiredCards) return;

    try {
      await gameStore.sendExchangeCards(cards);
      selectedIndices = [];
      gameStore.clearSelectedCards();
      gameStore.clearExchangePrompt();
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
    gameStore.clearExchangePrompt();
    gameStore.clearRoundSummary();
    gameStore.clearLogs();
    gameStore.clearJumpInPrompt();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(index: number) {
    if (isExchangeVisible && !exchangeCanChoose && exchangeRequiredCards > 0) {
      return;
    }

    const clickedCardValue = ownHand[index];
    const finishMove = gameStore.getAutoFinishMove();

    if (jumpInVisible && finishMove && clickedCardValue === finishMove[0]) {
      handleJumpIn();
      return;
    }

    if (selectedIndices.includes(index)) {
      selectedIndices = selectedIndices.filter((i) => i !== index);
    } else {
      const selectionLimit = isExchangeVisible ? exchangeRequiredCards : 4;
      if (selectedIndices.length >= selectionLimit) return;
      selectedIndices = [...selectedIndices, index];
    }
    gameStore.selectedCards.set(selectedIndices.map((i) => ownHand[i]));
  }

  function handleManualClear() {
    selectedIndices = [];
    gameStore.clearSelectedCards();
  }

  function syncVisualHand(backendHand: number[]) {
    if (visualHand.length === 0 || backendHand.length > visualHand.length) {
      const suits: ("clubs" | "diamonds" | "hearts" | "spades")[] = [
        "diamonds",
        "hearts",
        "spades",
      ];
      let suitIndex = 0;
      let forcedClubsGiven = false;

      visualHand = backendHand.map((value, idx) => {
        let assignedSuit: "clubs" | "diamonds" | "hearts" | "spades";
        if (
          value === 3 &&
          $gameState?.clubs_3_holder === 0 &&
          !forcedClubsGiven
        ) {
          assignedSuit = "clubs";
          forcedClubsGiven = true;
        } else {
          assignedSuit = suits[suitIndex % suits.length];
          suitIndex++;
        }
        return {
          id: `${value}-${idx}-${Math.random()}`,
          value,
          suit: assignedSuit,
        };
      });
    } else {
      const currentBackendCounts = backendHand.reduce(
        (acc, val) => {
          acc[val] = (acc[val] || 0) + 1;
          return acc;
        },
        {} as Record<number, number>,
      );
      const newVisualHand: VisualCard[] = [];
      for (const card of visualHand) {
        if (
          currentBackendCounts[card.value] &&
          currentBackendCounts[card.value] > 0
        ) {
          newVisualHand.push(card);
          currentBackendCounts[card.value]--;
        }
      }
      visualHand = newVisualHand;
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
      class="mx-auto grid min-h-[calc(100vh-2rem)] max-w-screen-2xl grid-rows-[auto_1fr_auto] gap-4 md:min-h-[calc(100vh-3rem)]"
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
        class="rounded-[2.5rem] border border-emerald-300/15 bg-[radial-gradient(circle_at_top,rgba(52,211,153,0.18),rgba(6,20,12,0.96)_68%)] px-5 py-5 shadow-[0_42px_120px_rgba(0,0,0,0.45)] backdrop-blur-md md:px-6 md:py-6"
      >
        <div
          class="grid gap-5 xl:grid-cols-[20rem_minmax(0,1.7fr)_20rem] xl:grid-rows-[auto_1fr]"
        >
          <div class="xl:col-span-3">
            {#if topOpponentViews.length > 0}
              <div class="flex flex-wrap justify-center gap-5 overflow-visible">
                {#each topOpponentViews as opponent}
                  <article
                    class="flex w-80 shrink-0 flex-col rounded-4xl border border-white/10 bg-black/25 px-4 py-4 shadow-lg shadow-black/20 backdrop-blur-md"
                  >
                    <div class="flex items-center gap-3">
                      <img
                        src="/bot.svg"
                        alt="Bot profile icon"
                        class="h-10 w-10 rounded-2xl border border-white/10 bg-white/10 p-2"
                      />
                      <div class="min-w-0">
                        <div
                          class="text-[0.65rem] uppercase tracking-[0.35em] text-green-200/70"
                        >
                          Seat {opponent.seat}
                        </div>
                        <div class="truncate text-lg font-black text-white">
                          {getOpponentLabel(opponent.role)} Bot
                        </div>
                      </div>
                    </div>

                    <div
                      class="mt-2 flex items-center justify-between text-xs uppercase tracking-[0.3em] text-green-100/65"
                    >
                      <span>Hand</span>
                      <span>{opponent.handCount} cards</span>
                    </div>

                    <div
                      class="flex w-full items-end justify-center overflow-visible"
                    >
                      {#each Array.from( { length: opponent.handCount }, ) as _, cardIndex}
                        <div
                          class="relative"
                          style={`margin-left: ${cardIndex === 0 ? 0 : -5.1}rem; z-index: ${cardIndex};`}
                        >
                          <Card
                            value={cardIndex + 1}
                            isFaceUp={false}
                            disabled={true}
                            className="shrink-0 scale-[0.5] origin-bottom md:scale-[0.56]"
                          />
                        </div>
                      {/each}
                    </div>
                  </article>
                {/each}
              </div>
            {/if}
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if leftOpponentView}
              <div
                class="flex w-full min-h-84 max-w-72 flex-col rounded-4xl border border-white/10 bg-black/25 p-4 shadow-2xl shadow-black/25 backdrop-blur-md"
              >
                <div class="flex items-center gap-3">
                  <img
                    src="/bot.svg"
                    alt="Bot profile icon"
                    class="h-10 w-10 rounded-2xl border border-white/10 bg-white/10 p-2"
                  />
                  <div class="min-w-0">
                    <div
                      class="text-[0.65rem] uppercase tracking-[0.35em] text-green-200/70"
                    >
                      Left
                    </div>
                    <div class="truncate text-lg font-black text-white">
                      {getOpponentLabel(leftOpponentView.role)} Bot
                    </div>
                  </div>
                </div>

                <div
                  class="mt-2 flex items-center justify-between text-xs uppercase tracking-[0.3em] text-green-100/65"
                >
                  <span>Seat {leftOpponentView.seat}</span>
                  <span>{leftOpponentView.handCount} cards</span>
                </div>

                <div
                  class="flex w-full items-end justify-center overflow-visible"
                >
                  {#each Array.from( { length: leftOpponentView.handCount }, ) as _, cardIndex}
                    <div
                      class="relative"
                      style={`margin-left: ${cardIndex === 0 ? 0 : -5.1}rem; z-index: ${cardIndex};`}
                    >
                      <Card
                        value={cardIndex + 1}
                        isFaceUp={false}
                        disabled={true}
                        className="shrink-0 scale-[0.5] origin-bottom md:scale-[0.56]"
                      />
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </aside>

          <div class="flex items-center justify-center xl:row-start-2">
            <div
              class="w-full max-w-2xl rounded-[2.5rem] border border-emerald-300/20 bg-[radial-gradient(circle_at_top,rgba(34,197,94,0.24),rgba(7,23,12,0.92)_70%)] p-6 shadow-[0_40px_100px_rgba(0,0,0,0.45)] backdrop-blur-md"
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
                  <div class="flex flex-col items-center gap-3">
                    <div
                      class="flex items-end justify-center overflow-visible"
                    >
                      {#each Array.from({ length: count }) as _, cardIndex}
                        <div
                          class="relative"
                          style={`margin-left: ${cardIndex === 0 ? 0 : -4}rem; z-index: ${cardIndex};`}
                        >
                          <Card
                            value={cardIndex >= count - twosUsed ? 15 : cardValue}
                            isFaceUp={true}
                            disabled={true}
                            className="shrink-0 scale-[0.9] origin-bottom"
                          />
                        </div>
                      {/each}
                    </div>
                    <div class="text-center text-xs uppercase tracking-[0.4em] text-emerald-200/70">
                      Current move to beat
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

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if rightOpponentView}
              <div
                class="flex w-full min-h-84 max-w-72 flex-col rounded-4xl border border-white/10 bg-black/25 p-4 shadow-2xl shadow-black/25 backdrop-blur-md"
              >
                <div class="flex items-center gap-3">
                  <img
                    src="/bot.svg"
                    alt="Bot profile icon"
                    class="h-10 w-10 rounded-2xl border border-white/10 bg-white/10 p-2"
                  />
                  <div class="min-w-0">
                    <div
                      class="text-[0.65rem] uppercase tracking-[0.35em] text-green-200/70"
                    >
                      Right
                    </div>
                    <div class="truncate text-lg font-black text-white">
                      {getOpponentLabel(rightOpponentView.role)} Bot
                    </div>
                  </div>
                </div>

                <div
                  class="mt-2 flex items-center justify-between text-xs uppercase tracking-[0.3em] text-green-100/65"
                >
                  <span>Seat {rightOpponentView.seat}</span>
                  <span>{rightOpponentView.handCount} cards</span>
                </div>

                <div
                  class="flex w-full items-end justify-center overflow-visible"
                >
                  {#each Array.from( { length: rightOpponentView.handCount }, ) as _, cardIndex}
                    <div
                      class="relative"
                      style={`margin-left: ${cardIndex === 0 ? 0 : -5.1}rem; z-index: ${cardIndex};`}
                    >
                      <Card
                        value={cardIndex + 1}
                        isFaceUp={false}
                        disabled={true}
                        className="shrink-0 scale-[0.5] origin-bottom md:scale-[0.56]"
                      />
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </aside>
        </div>
      </section>

      <footer
        class="rounded-3xl border border-white/10 bg-black/25 p-4 backdrop-blur-md md:p-6"
      >
        <div class="grid gap-4">
          {#if jumpInVisible}
            <div class="rounded-2xl border border-red-400/30 bg-red-500/10 p-4">
              <div
                class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between"
              >
                <div>
                  <p
                    class="text-xs uppercase tracking-[0.35em] text-red-200/80"
                  >
                    Jump-in window
                  </p>
                  <h2 class="mt-1 text-lg font-black text-red-50">
                    A finishing move is available
                  </h2>
                </div>
                <button
                  class="jump-in-button rounded-2xl border border-red-200/40 bg-red-500 px-5 py-3 text-lg font-black tracking-[0.25em] text-white shadow-lg shadow-red-950/30 transition hover:bg-red-400 active:scale-[0.99]"
                  style={`--jump-in-shake: ${2 + 10 * jumpInShake.current}; --jump-in-tilt: ${0.6 + 2.2 * jumpInShake.current};`}
                  type="button"
                  on:click={handleJumpIn}
                >
                  JUMP IN
                </button>
              </div>
            </div>
          {/if}

          <div
            class="rounded-4xl border border-white/10 bg-black/20 px-4 pb-6 pt-5 overflow-visible"
          >
            {#if ownHand.length === 0}
              <div
                class="mt-5 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-green-50/70"
              >
                No hand data yet
              </div>
            {:else}
              <div
                class="mt-5 flex w-full justify-center overflow-visible pb-5 pt-1"
              >
                {#each ownHand as card, index}
                  <div
                    class="relative"
                    style={getHorizontalOverlapStyle(index)}
                  >
                    <Card
                      value={card}
                      suit={visualHand[index]?.suit ?? "clubs"}
                      isFaceUp={true}
                      isSelected={selectedIndices.includes(index)}
                      disabled={isExchangeVisible &&
                        (!exchangeCanChoose || exchangeRequiredCards === 0)}
                      className="shrink-0 scale-[0.98]"
                      onClick={() => handleToggleCard(index)}
                    />
                  </div>
                {/each}
              </div>
            {/if}
          </div>
          <div
            class="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 p-4"
          >
            <div class="text-sm text-green-50/75">{selectedMoveLabel}</div>
            <div class="flex flex-wrap gap-3">
              {#if isExchangeVisible}
                <button
                  class={`rounded-xl px-5 py-3 font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
                    selectedHandCards.length === exchangeRequiredCards
                      ? "border border-emerald-200/30 bg-emerald-300 text-emerald-950 hover:bg-emerald-200"
                      : "border border-emerald-300/20 bg-emerald-500/40 text-emerald-50 hover:bg-emerald-400/50"
                  }`}
                  type="button"
                  disabled={selectedHandCards.length !== exchangeRequiredCards}
                  on:click={handleConfirmExchange}
                >
                  Confirm Exchange
                </button>
              {:else}
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
              {/if}
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

<style>
  @keyframes jump-in-shake {
    0% {
      transform: translateX(0) rotate(0deg);
    }
    20% {
      transform: translateX(calc(var(--jump-in-shake) * -1px))
        rotate(calc(var(--jump-in-tilt) * -1deg));
    }
    40% {
      transform: translateX(calc(var(--jump-in-shake) * 0.8px))
        rotate(calc(var(--jump-in-tilt) * 0.7deg));
    }
    60% {
      transform: translateX(calc(var(--jump-in-shake) * -0.6px))
        rotate(calc(var(--jump-in-tilt) * -0.45deg));
    }
    80% {
      transform: translateX(calc(var(--jump-in-shake) * 0.4px))
        rotate(calc(var(--jump-in-tilt) * 0.3deg));
    }
    100% {
      transform: translateX(0) rotate(0deg);
    }
  }

  .jump-in-button {
    animation: jump-in-shake 140ms linear infinite;
    will-change: transform;
  }
</style>
