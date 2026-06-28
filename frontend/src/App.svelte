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
    revealedBotSeat,
    selectedCards,
    state as gameState,
  } from "./stores/gameStore";
  import { send, receive } from "./lib/transitions";
  import { fly } from "svelte/transition";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";

  let gameStarted = false;
  let showFinalResults = false;

  $: suitedHand = $gameState?.suited_hand ?? [];
  $: lastMoveSuits = $gameState?.suit_last_move ?? [];
  $: totalPlayers = Object.keys($gameState?.player_roles ?? {}).length || 4;
  $: exchangeRequiredCards = $exchangePrompt?.requiredCards ?? 0;
  $: exchangeCanChoose = $exchangePrompt?.canChoose ?? false;
  $: isMyTurn = $gameState?.curr_turn === 0;
  $: haveIPassed = ($gameState?.passed?.includes(0) && !isMyTurn) ?? false;
  $: jumpInVisible = !!$jumpInPrompt && !!$gameState?.is_finish_prompt;
  $: jumpInAutoMove =
    jumpInVisible && $gameState ? gameStore.getAutoFinishMove() : null;
  $: jumpInTargetValue = jumpInAutoMove?.[0] ? jumpInAutoMove[0] : null;
  $: selectedMoveValues = $selectedCards.map(stripSuitCard);
  $: isSelectionLegal = (() => {
    if (!$gameState?.legal_moves_suits || !!$exchangePrompt) return false;
    const sortedSelection = [...selectedMoveValues].sort();
    return $gameState.legal_moves_suits.some((legalMove) =>
      arraysEqual([...legalMove].sort(), sortedSelection),
    );
  })();

  $: opponentViews = Object.entries($gameState?.player_roles ?? {})
    .map(([seatStr, role]) => {
      const seat = Number(seatStr);
      const relativeOffset = (seat + totalPlayers) % totalPlayers;
      return { seat, role, relativeOffset };
    })
    .filter((opponent) => opponent.seat !== 0)
    .sort((a, b) => a.relativeOffset - b.relativeOffset)
    .map((opponent, index, arr) => ({
      ...opponent,
      suitedHand: $gameState?.opp_suited_hands?.[opponent.seat] ?? [],
      label: `${displayBotType(opponent.role)} ${$gameState?.player_types?.[opponent.seat] ?? "Bot"}`,
      position:
        index === 0 ? "left" : index === arr.length - 1 ? "right" : "top",
      is_turn:
        $gameState?.curr_turn === opponent.seat &&
        !$gameState?.is_finish_prompt,
      has_passed: $gameState?.passed?.includes(opponent.seat) ?? false,
    }));

  $: topOpponents = opponentViews.filter((o) => o.position === "top");
  $: leftOpponent = opponentViews.find((o) => o.position === "left");
  $: rightOpponent = opponentViews.find((o) => o.position === "right");
  $: selectedMoveLabel = getMoveLabel(
    $selectedCards,
    !!$exchangePrompt,
    isMyTurn,
    isSelectionLegal,
    exchangeRequiredCards,
    exchangeCanChoose,
  );

  let lastStateRef: typeof $gameState = null;
  $: if ($gameState && !$exchangePrompt) {
    if ($gameState !== lastStateRef) {
      lastStateRef = $gameState;
      gameStore.clearSelectedCards();
    }
  }

  let lastExchangeActive = false;
  $: if (!!$exchangePrompt !== lastExchangeActive) {
    lastExchangeActive = !!$exchangePrompt;
    if (!!$exchangePrompt) {
      if (exchangeCanChoose) gameStore.clearSelectedCards();
      else if (exchangeRequiredCards > 0 && $gameState)
        gameStore.selectedCards.set(
          $gameState.suited_hand.slice(-exchangeRequiredCards),
        );
    }
  }

  onMount(() => {
    gameStore.connect(websocketUrl);
  });

  onDestroy(() => {
    gameStore.disconnect();
  });

  function getMoveLabel(
    cards: string[],
    exchange: boolean,
    myTurn: boolean,
    legal: boolean,
    reqCards: number,
    canChoose: boolean,
  ) {
    if (cards.length > 0) {
      if (exchange) return `Selected ${cards.length}/${reqCards} cards`;
      return legal ? `Valid Move: ${cards.length} cards` : "Illegal Move";
    }
    if (!exchange && !myTurn) return "Waiting for your turn";
    if (exchange) {
      if (reqCards === 0) return "Citizen's are exempt from exchanging";
      return canChoose
        ? `Select ${reqCards} cards to give away`
        : `Highest ${reqCards} cards pre-selected`;
    }
    return "Click cards from your hand to queue them for play";
  }

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

  function arraysEqual(left: string[], right: string[]) {
    if (left.length !== right.length) return false;
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
    if (!$gameState || !$exchangePrompt) return;

    const suitCards = $selectedCards;
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
    gameStore.clearSelectedCards();
    gameStore.clearExchangePrompt();
    gameStore.disconnect();
  }

  function handleBackToLobby() {
    showFinalResults = false;
    gameStarted = false;
    gameStore.clearAll();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(suitCard: string) {
    if (!!$exchangePrompt && !exchangeCanChoose && exchangeRequiredCards > 0)
      return;
    if (!$exchangePrompt && !isMyTurn) return;

    const clickedCardValue = stripSuitCard(suitCard);
    const finishMove = gameStore.getAutoFinishMove();

    if (
      jumpInVisible &&
      finishMove &&
      clickedCardValue !== null &&
      clickedCardValue === finishMove[0]
    ) {
      handleJumpIn();
      return;
    }

    if ($selectedCards.includes(suitCard)) {
      gameStore.selectedCards.set(
        $selectedCards.filter((selected) => selected !== suitCard),
      );
      return;
    }

    const selectionLimit = !!$exchangePrompt ? exchangeRequiredCards : 4;
    if ($selectedCards.length >= selectionLimit) return;
    gameStore.selectedCards.set([...$selectedCards, suitCard]);
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
      on:click={handleBackToLobby}
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
            {#if topOpponents.length > 0}
              <div class="flex flex-wrap justify-center gap-3 overflow-visible">
                {#each topOpponents as opponent (opponent.seat)}
                  <OpponentSeat
                    {opponent}
                    revealBotCards={$revealedBotSeat === opponent.seat}
                    className="w-60 shrink-0 px-3 py-1.5"
                  />
                {/each}
              </div>
            {/if}
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if leftOpponent}
              <OpponentSeat
                opponent={leftOpponent}
                revealBotCards={$revealedBotSeat === leftOpponent.seat}
                className="w-full max-w-56 p-3"
              />
            {/if}
          </aside>

          <div class="flex items-center justify-center xl:row-start-2">
            <div
              class="w-full max-w-md flex items-center justify-center p-3 min-h-28 relative"
            >
              <div class="flex items-end justify-center overflow-visible h-20">
                {#each lastMoveSuits as suitCard, cardIndex (suitCard)}
                  <div
                    in:receive={{ key: suitCard }}
                    out:send={{ key: suitCard }}
                    on:introstart={gameStore.startAnimation}
                    on:introend={gameStore.endAnimation}
                    on:outrostart={gameStore.startAnimation}
                    on:outroend={gameStore.endAnimation}
                    class="relative transition-all duration-300"
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
              {#if lastMoveSuits.length === 0}
                <div
                  transition:fly={{ y: 10, duration: 200 }}
                  class="absolute text-center text-emerald-50/70 py-1 pointer-events-none"
                >
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
            {#if rightOpponent}
              <OpponentSeat
                opponent={rightOpponent}
                revealBotCards={$revealedBotSeat === rightOpponent.seat}
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
                class="my-0.5 flex w-full justify-center overflow-visible py-1 h-20 items-center transition-all duration-300 {haveIPassed
                  ? 'opacity-40 grayscale pointer-events-none'
                  : ''}"
              >
                {#each suitedHand as suitCard, index (suitCard)}
                  <div
                    in:receive={{ key: suitCard }}
                    out:send={{ key: suitCard }}
                    on:introstart={gameStore.startAnimation}
                    on:introend={gameStore.endAnimation}
                    on:outrostart={gameStore.startAnimation}
                    on:outroend={gameStore.endAnimation}
                    class="relative pointer-events-none transition-all duration-300"
                    style={getHorizontalOverlapStyle(index)}
                  >
                    <Card
                      {suitCard}
                      isFaceUp={true}
                      isSelected={$selectedCards.includes(suitCard)}
                      isBlinking={jumpInVisible &&
                        jumpInTargetValue === stripSuitCard(suitCard)}
                      disabled={haveIPassed ||
                        (!$exchangePrompt && !isMyTurn) ||
                        (!!$exchangePrompt &&
                          (!exchangeCanChoose || exchangeRequiredCards === 0))}
                      className="shrink-0 scale-[0.75] md:scale-[0.8] pointer-events-auto"
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
              {#if !!$exchangePrompt}
                <button
                  class={`rounded-lg px-4 py-1.5 font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 text-xs ${
                    $selectedCards.length === exchangeRequiredCards
                      ? "border border-emerald-200/30 bg-emerald-300 text-emerald-950 hover:bg-emerald-200"
                      : "border border-emerald-300/20 bg-emerald-500/40 text-emerald-50 hover:bg-emerald-400/50"
                  }`}
                  type="button"
                  disabled={$selectedCards.length !== exchangeRequiredCards}
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
                  disabled={!isMyTurn || !$gameState?.can_pass}
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
          scores={$roundSummary ? Object.entries($roundSummary.scores) : []}
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
      {:else if $roundSummary}
        <SummaryModal
          eyebrow="Round complete"
          eyebrowClass="text-amber-200/75"
          title="Play the next round or quit"
          scores={Object.entries($roundSummary.scores)}
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
