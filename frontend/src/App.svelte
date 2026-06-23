<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import Card from "./assets/Card.svelte";
  import Lobby from "./lib/components/Lobby.svelte";
  import OpponentSeat from "./lib/components/OpponentSeat.svelte";
  import SummaryModal from "./lib/components/SummaryModal.svelte";
  import {
    gameStore,
    exchangePrompt,
    jumpInPrompt,
    roundSummary,
    selectedCards,
    state as gameState,
  } from "./stores/gameStore";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";

  let gameStarted = false;
  let showFinalResults = false;

  $: currentState = $gameState;
  $: exchangePromptData = $exchangePrompt;
  $: jumpInPromptData = $jumpInPrompt;
  $: roundSummaryData = $roundSummary;
  $: suitedHand = currentState?.suited_hand ?? [];
  $: lastMoveSuits = currentState?.suit_last_move ?? [];
  $: totalPlayers = Object.keys(currentState?.player_roles ?? {}).length || 4;
  $: selectedHandCards = $selectedCards;
  $: exchangeRequiredCards = exchangePromptData?.requiredCards ?? 0;
  $: exchangeCanChoose = exchangePromptData?.canChoose ?? false;
  $: isExchangeVisible = !!exchangePromptData;
  $: isMyTurn = currentState?.curr_turn === 0;
  $: jumpInVisible = !!jumpInPromptData && !!currentState?.is_finish_prompt;
  $: jumpInAutoMove = jumpInVisible && currentState
    ? gameStore.getAutoFinishMove()
    : null;
  $: jumpInTargetValue = jumpInAutoMove?.[0]
    ? parseSuitValue(jumpInAutoMove[0])
    : null;
  $: selectedMoveValues = selectedHandCards.map(stripSuitCard);
  $: isSelectionLegal = (() => {
    if (!currentState?.legal_moves_suits || isExchangeVisible) return false;
    return currentState.legal_moves_suits.some((legalMove) =>
      arraysEqual(legalMove, selectedMoveValues),
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
  $: opponentViews = opponentSeats.map((opponent, index) => ({
    ...opponent,
    handCount: currentState?.opp_hand_counts?.[opponent.seat] ?? 0,
    label: `${displayBotType(opponent.role)} ${currentState?.player_types?.[opponent.seat] ?? "Bot"}`,
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
  $: selectedMoveLabel = selectedHandCards.length > 0
    ? isExchangeVisible
      ? `Selected ${selectedHandCards.length}/${exchangeRequiredCards} cards`
      : isSelectionLegal
        ? `Valid Move: ${selectedHandCards.length} cards`
        : "Illegal Move"
    : !isExchangeVisible && !isMyTurn
      ? "Waiting for your turn"
      : isExchangeVisible
        ? exchangeRequiredCards === 0
          ? "Citizen's are exempt from exchanging"
          : exchangeCanChoose
            ? `Select ${exchangeRequiredCards} cards to give away`
            : `Highest ${exchangeRequiredCards} cards pre-selected`
        : "Click cards from your hand to queue them for play";
  let lastStateRef: typeof currentState = null;
  $: if (currentState && !isExchangeVisible) {
    if (currentState !== lastStateRef) {
      lastStateRef = currentState;
      gameStore.clearSelectedCards();
    }
  }

  let lastExchangeActive = false;
  $: if (isExchangeVisible !== lastExchangeActive) {
    lastExchangeActive = isExchangeVisible;
    if (isExchangeVisible) {
      if (exchangeCanChoose) {
        gameStore.clearSelectedCards();
      } else if (exchangeRequiredCards > 0 && currentState) {
        gameStore.selectedCards.set(
          currentState.suited_hand.slice(-exchangeRequiredCards),
        );
      }
    }
  }

  onMount(() => {
    gameStore.connect(websocketUrl);
  });

  onDestroy(() => {
    gameStore.disconnect();
  });

  function displayBotType(role: string | null) {
    if (!role) return "Waiting";
    return role.replace(/\s*Bot$/i, "");
  }

  function getHorizontalOverlapStyle(index: number) {
    return index === 0 ? "" : "margin-left: -4rem;";
  }

  function stripSuitCard(suitCard: string) {
    return suitCard.slice(0, -1).toUpperCase();
  }

  function parseSuitValue(suitCard: string) {
    const rankText = stripSuitCard(suitCard);
    if (rankText === "T") return 10;
    if (rankText === "J") return 11;
    if (rankText === "Q") return 12;
    if (rankText === "K") return 13;
    if (rankText === "A") return 14;
    if (rankText === "2") return 15;

    const parsed = Number.parseInt(rankText, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function arraysEqual(left: string[], right: string[]) {
    if (left.length !== right.length) {
      return false;
    }

    return left.every((value, index) => value === right[index]);
  }

  async function handleStartGame(config: {
    numPlayers: number;
    numRounds: number;
    playerTypes: number[];
  }) {
    const payload: {
      type: "START_GAME";
      num_players: number;
      num_rounds: number;
      player_types: number[];
    } = {
      type: "START_GAME",
      num_players: config.numPlayers,
      num_rounds: config.numRounds,
      player_types: config.playerTypes,
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

    const suitCards = selectedHandCards;
    if (suitCards.length !== exchangeRequiredCards) return;

    try {
      await gameStore.sendExchangeCards(suitCards);
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
    gameStore.clearSelectedCards();
    gameStore.clearExchangePrompt();
    gameStore.clearRoundSummary();
    gameStore.clearLogs();
    gameStore.clearJumpInPrompt();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(suitCard: string) {
    if (isExchangeVisible && !exchangeCanChoose && exchangeRequiredCards > 0) {
      return;
    }

    if (!isExchangeVisible && !isMyTurn) {
      return;
    }

    const clickedCardValue = parseSuitValue(suitCard);
    const finishMove = gameStore.getAutoFinishMove();

    if (
      jumpInVisible &&
      finishMove &&
      clickedCardValue !== null &&
      clickedCardValue === parseSuitValue(finishMove[0])
    ) {
      handleJumpIn();
      return;
    }

    if (selectedHandCards.includes(suitCard)) {
      gameStore.selectedCards.set(
        selectedHandCards.filter((selected) => selected !== suitCard),
      );
      return;
    }

    const selectionLimit = isExchangeVisible ? exchangeRequiredCards : 4;
    if (selectedHandCards.length >= selectionLimit) return;
    gameStore.selectedCards.set([...selectedHandCards, suitCard]);
  }
</script>

{#if !gameStarted}
  <Lobby onStartGame={handleStartGame} />
{:else}
  <main
    class="h-screen max-h-screen overflow-hidden bg-[radial-gradient(circle_at_center,#154d2a_0%,#0b2414_48%,#050b07_100%)] p-2 text-white md:p-3"
  >
    <button
      class="fixed right-4 top-4 z-30 rounded-md border border-red-300/30 bg-red-500 px-3 py-1 text-[0.65rem] font-black tracking-[0.28em] text-white shadow-lg shadow-black/30 transition hover:bg-red-400 active:scale-[0.98]"
      type="button"
      on:click={handleQuit}
    >
      QUIT
    </button>
    <section
      class="mx-auto grid h-full max-h-full grid-rows-[1fr_auto] gap-2 max-w-screen-2xl"
    >
      <section
        class="rounded-2xl border border-emerald-300/15 bg-[radial-gradient(circle_at_top,rgba(52,211,153,0.18),rgba(6,20,12,0.96)_68%)] p-3 shadow-lg backdrop-blur-md flex flex-col justify-between overflow-hidden"
      >
        <div
          class="grid gap-3 xl:grid-cols-[16rem_1fr_16rem] xl:grid-rows-[auto_1fr] h-full items-center"
        >
          <div class="xl:col-span-3">
            {#if topOpponentViews.length > 0}
              <div class="flex flex-wrap justify-center gap-3 overflow-visible">
                {#each topOpponentViews as opponent}
                  <OpponentSeat
                    {opponent}
                    className="w-60 shrink-0 px-3 py-1.5"
                  />
                {/each}
              </div>
            {/if}
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if leftOpponentView}
              <OpponentSeat
                opponent={leftOpponentView}
                className="w-full max-w-56 p-3"
              />
            {/if}
          </aside>

          <div class="flex items-center justify-center xl:row-start-2">
            <div
              class="w-full max-w-md flex items-center justify-center p-3 min-h-28"
            >
              {#if lastMoveSuits.length > 0}
                <div class="flex flex-col items-center gap-1">
                  <div
                    class="flex items-end justify-center overflow-visible h-20"
                  >
                    {#each lastMoveSuits as suitCard, cardIndex}
                      <div
                        class="relative"
                        style={`margin-left: ${cardIndex === 0 ? 0 : -4.3}rem; z-index: ${cardIndex};`}
                      >
                        <Card
                          {suitCard}
                          isFaceUp={true}
                          disabled={true}
                          className="shrink-0 scale-[0.7] origin-bottom"
                        />
                      </div>
                    {/each}
                  </div>
                </div>
              {:else}
                <div class="text-center text-emerald-50/70 py-1">
                  <div class="text-[0.6rem] uppercase tracking-[0.3em]">
                    Game Pile
                  </div>
                  <div class="mt-0.5 text-xl font-black text-white/70">
                    Empty
                  </div>
                  <div class="mt-0.5 text-xs">
                    Play any valid combination to lead
                  </div>
                </div>
              {/if}
            </div>
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if rightOpponentView}
              <OpponentSeat
                opponent={rightOpponentView}
                className="w-full max-w-56 p-3"
              />
            {/if}
          </aside>
        </div>
        <div class="grid gap-2">
          <div class="px-3 py-1 overflow-visible">
            {#if suitedHand.length === 0}
              <div
                class="my-1 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-green-50/70 text-center"
              >
                No hand data yet
              </div>
            {:else}
              <div
                class="my-0.5 flex w-full justify-center overflow-visible py-1 h-20 items-center"
              >
                {#each suitedHand as suitCard, index}
                  <div
                    class="relative"
                    style={getHorizontalOverlapStyle(index)}
                  >
                    <Card
                      {suitCard}
                      isFaceUp={true}
                      isSelected={selectedHandCards.includes(suitCard)}
                      isBlinking={jumpInVisible &&
                        jumpInTargetValue === parseSuitValue(suitCard)}
                      disabled={(!isExchangeVisible && !isMyTurn) ||
                        (isExchangeVisible &&
                          (!exchangeCanChoose || exchangeRequiredCards === 0))}
                      className="shrink-0 scale-[0.75] md:scale-[0.8]"
                      onClick={() => handleToggleCard(suitCard)}
                    />
                  </div>
                {/each}
              </div>
            {/if}
          </div>
          <div
            class="flex items-center justify-between gap-2 rounded-xl border border-white/10 bg-white/5 p-2 text-xs md:text-sm"
          >
            <div class="text-xs text-green-50/75">{selectedMoveLabel}</div>
            <div class="flex gap-2">
              {#if isExchangeVisible}
                <button
                  class={`rounded-lg px-4 py-1.5 font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 text-xs ${
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
                  class="rounded-lg border border-emerald-300/30 bg-emerald-400 px-4 py-1.5 font-semibold text-emerald-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-40 text-xs"
                  type="button"
                  disabled={!isMyTurn || !isSelectionLegal}
                  on:click={handlePlay}
                >
                  Play
                </button>
                <button
                  class="rounded-lg border border-white/10 bg-white/10 px-4 py-1.5 font-semibold text-white transition hover:bg-white/15 text-xs"
                  type="button"
                  disabled={!isMyTurn || !currentState?.can_pass}
                  on:click={handlePass}
                >
                  Pass
                </button>
              {/if}
            </div>
          </div>
        </div>
      </section>

      {#if showFinalResults}
        <SummaryModal
          eyebrow="Game ended"
          eyebrowClass="text-emerald-300/80"
          title="Final Results"
          scores={roundSummaryData
            ? Object.entries(roundSummaryData.scores)
            : []}
        >
          <button
            slot="actions"
            class="rounded-xl border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
            type="button"
            on:click={handleBackToLobby}
          >
            Back to Lobby
          </button>
        </SummaryModal>
      {:else if roundSummaryData}
        <SummaryModal
          eyebrow="Round complete"
          eyebrowClass="text-amber-200/75"
          title="Play the next round or quit"
          scores={Object.entries(roundSummaryData.scores)}
        >
          <div slot="actions" class="flex gap-3">
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
        </SummaryModal>
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
