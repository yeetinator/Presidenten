<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import Card from "./assets/Card.svelte";
  import Lobby from "./lib/components/Lobby.svelte";
  import Rules from "./assets/Rules.svelte";
  import OpponentSeat from "./lib/components/OpponentSeat.svelte";
  import SummaryModal from "./lib/components/SummaryModal.svelte";
  import {
    gameStore,
    exchangePrompt,
    jumpInPrompt,
    fastForwardMode,
    roundSummary,
    revealedBotSeat,
    selectedCards,
    enableCards,
    state as gameState,
  } from "./stores/gameStore";
  import { send, receive } from "./lib/transitions";
  import { fade, fly } from "svelte/transition";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";

  let gameStarted = false;
  let transitionFlyDuration = 200;

  $: transitionFlyDuration = $fastForwardMode ? 100 : 200;

  $: suitedHand = $gameState?.suited_hand ?? [];
  $: totalPlayers = Object.keys($gameState?.player_roles ?? {}).length || 4;
  $: exchangeRequiredCards = $exchangePrompt?.requiredCards ?? 0;
  $: exchangeCanChoose = $exchangePrompt?.canChoose ?? false;
  $: isMyTurn = $gameState?.curr_turn === 0;
  $: haveIPassed = ($gameState?.passed?.includes(0) && !isMyTurn) ?? false;
  $: jumpInVisible = !!$jumpInPrompt && !!$gameState?.is_finish_prompt;
  $: jumpInAutoMove =
    jumpInVisible && $gameState ? gameStore.getAutoFinishMove() : null;
  $: selectedMoveValues = $selectedCards.map(stripSuitCard);
  $: isSelectionLegal = (() => {
    if (
      !$gameState?.legal_moves_suits ||
      $exchangePrompt ||
      !$selectedCards.length ||
      $selectedCards.length === 0
    )
      return false;
    const sortedSelection = [...selectedMoveValues].sort();
    return $gameState.legal_moves_suits.some((legalMove) =>
      arraysEqual([...legalMove].sort(), sortedSelection),
    );
  })();

  $: isExchangeReady =
    !!$exchangePrompt && $selectedCards.length === exchangeRequiredCards;
  $: isCenterClickable = (isMyTurn && isSelectionLegal) || isExchangeReady;

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
        $gameState?.resume_turn === opponent.seat ||
        ($gameState?.curr_turn === opponent.seat && !$gameState?.resume_turn),
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
    !!$gameState?.can_pass,
    $gameState?.legal_moves_suits,
  );

  let visualPile: string[][] = [];
  let cardRotations: Record<string, number> = {};
  $: {
    const currSuits = $gameState?.suit_last_move ?? [];
    if (currSuits.length === 0) {
      visualPile = [];
      cardRotations = {};
    } else {
      const lastPlay = visualPile[visualPile.length - 1];
      const isAlreadyAdded =
        lastPlay &&
        lastPlay.length === currSuits.length &&
        currSuits.every((card, idx) => card === lastPlay[idx]);
      const isExtension =
        lastPlay &&
        currSuits.length > lastPlay.length &&
        lastPlay.every((card, idx) => card === currSuits[idx]);

      if (!isAlreadyAdded) {
        if (isExtension) {
          visualPile[visualPile.length - 1] = currSuits;
          visualPile = [...visualPile];
        } else visualPile = [...visualPile, currSuits];

        const leaderSeat = $gameState?.pile_leader ?? 0;
        let baseAngle = 0;

        if (leaderSeat !== 0 && opponentViews) {
          const throwingOpponent = opponentViews.find(
            (o) => o.seat === leaderSeat,
          );
          if (throwingOpponent) {
            if (throwingOpponent.position === "left") baseAngle = 90;
            else if (throwingOpponent.position === "top") baseAngle = 180;
            else if (throwingOpponent.position === "right") baseAngle = 270;
          }
        }
        const groupPlayJitter = Math.random() * 70 - 35;
        currSuits.forEach((suitCard) => {
          if (cardRotations[suitCard] === undefined) {
            const microJitter = Math.random() * 10 - 5;
            cardRotations[suitCard] = baseAngle + groupPlayJitter + microJitter;
          }
        });
      }
    }
  }
  $: flatVisualPile = visualPile.flatMap((play, playIndex) =>
    play.map((suitCard, cardIndex) => ({ suitCard, playIndex, cardIndex })),
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
    if ($exchangePrompt) {
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
    canPass: boolean,
    legalMoves: string[][] | undefined,
  ) {
    if (exchange) {
      if (cards.length > 0) return `Selected ${cards.length}/${reqCards} cards`;
      if (reqCards === 0) return "Citizen's are exempt from exchanging";
      return canChoose
        ? `Select ${reqCards} cards to give away`
        : `Highest ${reqCards} cards pre-selected`;
    }

    if (!myTurn) return "Waiting for your turn";

    if (cards.length > 0)
      return legal ? `Valid Move: ${cards.length} cards` : "Illegal Move";

    if (legalMoves && legalMoves.length === 0 && canPass)
      return "You can only pass this turn";

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
      gameStore.clearFastForwardMode();
      await gameStore.startGame(payload);
      gameStarted = true;
    } catch (error) {
      console.error(error);
    }
  }

  function handleCenterAction() {
    if (isExchangeReady) handleConfirmExchange();
    else if (isMyTurn && isSelectionLegal) handlePlay();
  }

  async function handlePlay() {
    try {
      console.log("Playing selected cards:", $selectedCards);
      $enableCards = false;
      await gameStore.playSelectedCards();
    } catch (error) {
      console.error(error);
    }
  }

  async function handlePass() {
    try {
      $enableCards = false;
      await gameStore.passTurn();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleJumpIn(finishMove: string[]) {
    try {
      $enableCards = false;
      await gameStore.playJumpInPrompt(finishMove);
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
      $enableCards = false;
      await gameStore.sendExchangeCards(suitCards);
      gameStore.clearSelectedCards();
      gameStore.clearExchangePrompt();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleFastForward() {
    try {
      fastForwardMode.set(true);
      await gameStore.fastForwardGame();
    } catch (error) {
      console.error(error);
    }
  }

  function handleQuit() {
    gameStarted = false;
    gameStore.clearAll();
    gameStore.disconnect();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(suitCard: string) {
    if ($exchangePrompt && !exchangeCanChoose && exchangeRequiredCards > 0)
      return;
    if (!$exchangePrompt && !isMyTurn) return;

    const clickedCardValue = stripSuitCard(suitCard);
    const finishMove = gameStore.getAutoFinishMove(clickedCardValue);

    if (jumpInVisible && finishMove && clickedCardValue !== null) {
      handleJumpIn(finishMove);
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
    class="h-screen max-h-screen overflow-hidden bg-[radial-gradient(circle_at_center,#154d2a_0%,#0b2414_48%,#050b07_100%)]
      p-2 text-white md:p-3"
  >
    {#if !$roundSummary}
      <Rules className="left-4" side="left" />
      <button
        class="fixed right-5 top-4 z-30 rounded-md border border-red-300/30 bg-red-500 px-3 py-1 text-[0.65rem]
        font-black tracking-[0.28em] text-white shadow-lg shadow-black/30 transition hover:bg-red-400 active:scale-[0.98]"
        type="button"
        on:click={handleQuit}
      >
        QUIT
      </button>
      {#if suitedHand.length === 0 && opponentViews.some((o) => o.suitedHand.length !== 0) && !$exchangePrompt}
        <button
          class="fixed right-22 top-4 z-30 rounded-md border border-blue-300/30 bg-blue-500 px-3 py-1 text-[0.65rem]
        font-black tracking-[0.28em] text-white shadow-lg shadow-black/30 transition hover:bg-blue-400 active:scale-[0.98]"
          type="button"
          on:click={handleFastForward}
        >
          {#if $fastForwardMode}
            <div in:fade={{ duration: 150 }} out:fade={{ duration: 150 }}>
              <span class="animate-arrow-pulse opacity-20">&gt;</span>
              <span
                class="animate-arrow-pulse opacity-20 [animation-delay:0.15s]"
                >&gt;</span
              >
              <span
                class="animate-arrow-pulse opacity-20 [animation-delay:0.3s]"
                >&gt;</span
              >
            </div>
          {:else}
            <span in:fade={{ duration: 150 }} out:fade={{ duration: 150 }}
              >FAST FORWARD</span
            >
          {/if}
        </button>
      {/if}
    {/if}
    <section
      class="mx-auto grid h-full max-h-full grid-rows-[1fr_auto] gap-2 max-w-screen-2xl"
    >
      <section
        class="rounded-2xl border border-emerald-300/15 bg-[radial-gradient(circle_at_top,rgba(52,211,153,0.18),rgba(6,20,12,0.96)_68%)]
          p-1 shadow-lg backdrop-blur-md flex flex-col justify-between overflow-hidden"
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
            <button
              type="button"
              class="w-full max-w-xs aspect-square flex flex-col justify-between p-4 relative rounded-3xl border transition-all duration-300 outline-none
              {isExchangeReady
                ? 'border-amber-400 bg-amber-500/5 shadow-[0_0_30px_rgba(245,158,11,0.25)] hover:bg-amber-500/10 cursor-pointer active:scale-[0.99]'
                : isMyTurn && isSelectionLegal
                  ? 'border-emerald-400 bg-emerald-500/5 shadow-[0_0_30px_rgba(52,211,153,0.2)] hover:bg-emerald-500/10 cursor-pointer active:scale-[0.99]'
                  : 'border-white/5 bg-white/1 cursor-default'}"
              disabled={!isCenterClickable}
              on:click={handleCenterAction}
            >
              <div
                class="w-full text-center pointer-events-none z-10 select-none"
              >
                {#if isExchangeReady}
                  <span
                    class="text-[0.6rem] font-black uppercase tracking-[0.25em] text-amber-400 animate-pulse"
                    >➔ Click Here to Send Cards
                  </span>
                {:else if isMyTurn && isSelectionLegal}
                  <span
                    class="text-[0.6rem] font-black uppercase tracking-[0.25em] text-emerald-400 animate-pulse"
                    >➔ Click Here to Play Move
                  </span>
                {:else if $selectedCards.length > 0 && !isSelectionLegal && !$exchangePrompt}
                  <span
                    class="text-[0.6rem] font-bold uppercase tracking-[0.25em] text-red-400/90"
                    >✕ Invalid Combination</span
                  >
                {:else}
                  <span
                    class="text-[0.55rem] font-bold uppercase tracking-[0.25em] text-white/20"
                    >Game Pile</span
                  >
                {/if}
              </div>
              <div
                class="absolute inset-0 flex items-center justify-center overflow-visible pointer-events-none"
              >
                <div
                  class="relative flex items-center justify-center overflow-visible h-28 w-full"
                >
                  {#each flatVisualPile as card (card.suitCard)}
                    <div
                      in:receive={{ key: card.suitCard }}
                      out:send={{ key: card.suitCard, isPile: true }}
                      on:introstart={gameStore.startAnimation}
                      on:introend={gameStore.endAnimation}
                      on:outrostart={gameStore.startAnimation}
                      on:outroend={gameStore.endAnimation}
                      class="absolute transition-all duration-300 flex justify-center origin-center"
                      style={`transition-duration: ${$fastForwardMode ? "100ms" : "200ms"}; z-index: ${card.playIndex * 10 + card.cardIndex}; 
                      transform: translateX(${(card.cardIndex - (visualPile[card.playIndex].length - 1) / 2) * 1.4}rem);`}
                    >
                      <div
                        style={`transform: rotate(${cardRotations[card.suitCard] ?? 0}deg);`}
                        class="origin-center flex justify-center items-center"
                      >
                        <Card
                          suitCard={card.suitCard}
                          isFaceUp={true}
                          disabled={true}
                          className="shrink-0 scale-[0.85] origin-center shadow-lg shadow-black/40"
                        />
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
              <div
                class="w-full text-center pointer-events-none z-10 select-none text-[0.65rem] font-medium tracking-wide text-white/60 leading-normal max-w-56 mx-auto"
              >
                {selectedMoveLabel}
              </div>
            </button>
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
        <div class="w-full flex flex-col gap-1 mt-auto">
          <div class="h-9 flex justify-center z-20 overflow-visible">
            {#if isMyTurn && $gameState?.can_pass && !$exchangePrompt && $gameState?.pile_leader !== 0}
              <button
                transition:fade={{ duration: 100 }}
                type="button"
                class="rounded-full border border-white/20 bg-black/50 backdrop-blur-md px-6 py-1.5 text-[0.65rem] font-black uppercase tracking-[0.2em] text-white/90 shadow-md transition-all hover:bg-white/10 hover:text-white active:scale-95"
                on:click={handlePass}>Pass Turn</button
              >
            {/if}
          </div>
          <div
            class="px-3 pt-6 overflow-visible relative flex items-center justify-center"
          >
            <div
              class="flex w-full justify-center overflow-visible h-28 md:h-36 items-center transition-all duration-300 {haveIPassed
                ? 'opacity-40 grayscale pointer-events-none'
                : ''}"
              style={`transition-duration: ${$fastForwardMode ? "100ms" : "200ms"};`}
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
                  style={`transition-duration: ${$fastForwardMode ? "100ms" : "200ms"}; ${getHorizontalOverlapStyle(index)}`}
                >
                  <Card
                    {suitCard}
                    isFaceUp={true}
                    isSelected={$selectedCards.includes(suitCard)}
                    isBlinking={!!(
                      jumpInVisible &&
                      jumpInAutoMove?.some((arr) =>
                        arr.includes(stripSuitCard(suitCard)),
                      )
                    )}
                    disabled={haveIPassed ||
                      (!$exchangePrompt && !isMyTurn) ||
                      (!!$exchangePrompt &&
                        (!exchangeCanChoose || exchangeRequiredCards === 0)) ||
                      !$enableCards}
                    exchange={!!$exchangePrompt}
                    className="shrink-0 scale-[0.75] md:scale-[0.95] pointer-events-auto"
                    onClick={() => handleToggleCard(suitCard)}
                  />
                </div>
              {/each}
            </div>
          </div>
        </div>
      </section>

      {#if $roundSummary}
        <SummaryModal
          eyebrow="Round complete"
          eyebrowClass="text-amber-200/75"
          title="Play the next round or quit"
          scores={$roundSummary.scores}
          playerTypes={$gameState?.player_types}
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
  @keyframes arrow-pulse {
    0%,
    100% {
      opacity: 0.2;
      transform: translateX(0);
    }
    50% {
      opacity: 1;
      transform: translateX(0.2rem);
    }
  }
  :global(.animate-arrow-pulse) {
    animation: arrow-pulse 0.6s infinite ease-in-out;
  }
</style>
