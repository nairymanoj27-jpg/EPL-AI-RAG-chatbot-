/* Premier League RAG frontend.
 *
 * No frameworks and no third-party scripts. All server-provided text is
 * inserted with textContent so retrieved documents cannot inject markup.
 */
(() => {
  "use strict";

  const API_BASE = (
    document.querySelector('meta[name="api-base"]')?.getAttribute("content") || ""
  ).replace(/\/+$/, "");
  const STORAGE_KEY = "pl_rag_api_key";
  const REQUEST_TIMEOUT_MS = 90_000;

  const state = {
    conversationId: null,
    lastQuestion: null,
    apiKey: sessionStorage.getItem(STORAGE_KEY) || "",
    busy: false,
    requestSeq: 0,
  };

  const el = {
    welcome: document.getElementById("welcome"),
    messages: document.getElementById("messages"),
    composer: document.getElementById("composer"),
    input: document.getElementById("composer-input"),
    send: document.getElementById("send-button"),
    charCount: document.getElementById("char-count"),
    loading: document.getElementById("loading"),
    loadingText: document.getElementById("loading-text"),
    errorBanner: document.getElementById("error-banner"),
    errorTitle: document.getElementById("error-title"),
    errorMessage: document.getElementById("error-message"),
    retry: document.getElementById("retry-button"),
    dismissError: document.getElementById("dismiss-error"),
    reset: document.getElementById("reset-button"),
    settingsButton: document.getElementById("settings-button"),
    dialog: document.getElementById("settings-dialog"),
    settingsForm: document.getElementById("settings-form"),
    apiKeyInput: document.getElementById("api-key-input"),
    clearKey: document.getElementById("clear-key"),
    cancelKey: document.getElementById("cancel-key"),
    connection: document.getElementById("connection-status"),
    connectionText: document.getElementById("connection-text"),
  };

  class ApiError extends Error {
    constructor(message, status = 0, code = "error", details = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
      this.details = details;
    }
  }

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function scrollToBottom() {
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth";
    window.scrollTo({ top: document.body.scrollHeight, behavior });
  }

  function setConnection(mode, text) {
    el.connection.dataset.state = mode;
    el.connectionText.textContent = text;
  }

  async function api(path, { method = "GET", body, signal, timeout = REQUEST_TIMEOUT_MS } = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
    const onExternalAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener("abort", onExternalAbort, { once: true });
    }

    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (state.apiKey) headers["X-API-Key"] = state.apiKey;

    let response;
    try {
      response = await fetch(`${API_BASE}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        credentials: "omit",
        cache: "no-store",
      });
    } catch (error) {
      if (error && error.name === "AbortError") {
        if (signal && signal.aborted) {
          throw new ApiError("The request was cancelled.", 0, "cancelled");
        }
        throw new ApiError(
          "The request timed out. The model may still be working; try again in a moment.",
          0,
          "timeout"
        );
      }
      throw new ApiError(
        "Could not reach the API. Check that the server is running and the URL is correct.",
        0,
        "network"
      );
    } finally {
      window.clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onExternalAbort);
    }

    const raw = await response.text();
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = null;
      }
    }

    if (!response.ok) {
      const detail = data && data.error ? data.error : null;
      throw new ApiError(
        (detail && detail.message) || `The request failed (HTTP ${response.status}).`,
        response.status,
        (detail && detail.code) || "http_error",
        data && data.details ? data.details : null
      );
    }
    return data;
  }

  async function checkHealth() {
    setConnection("unknown", "Checking API…");
    try {
      const health = await api("/health", { timeout: 8_000 });
      const version = health && health.version ? `v${health.version}` : "";
      setConnection("ok", `API online${version ? ` · ${version}` : ""}`);
      el.connection.title = health && health.name ? health.name : "API online";
    } catch (error) {
      if (error.status === 401) {
        setConnection("locked", "API key required");
        el.connection.title = "Open the API key dialog to add your key.";
      } else {
        setConnection("error", "API unreachable");
        el.connection.title = error.message;
      }
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    el.send.disabled = busy;
    el.input.setAttribute("aria-busy", busy ? "true" : "false");
    el.messages.setAttribute("aria-busy", busy ? "true" : "false");
    el.loading.hidden = !busy;
    if (!busy) {
      el.loadingText.textContent = "Searching the index and drafting an answer…";
    }
  }

  function hideWelcome() {
    el.welcome.hidden = true;
  }

  function showWelcome() {
    el.welcome.hidden = false;
  }

  function appendUserMessage(text) {
    hideWelcome();
    const wrapper = make("article", "message user");
    wrapper.appendChild(make("div", "avatar", "You"));
    const body = make("div", "message-body");
    const bubble = make("div", "bubble");
    bubble.appendChild(make("div", "answer", text));
    body.appendChild(bubble);
    wrapper.appendChild(body);
    el.messages.appendChild(wrapper);
    scrollToBottom();
  }

  function renderCitations(text, sourceIds) {
    const fragment = document.createDocumentFragment();
    const pattern = /\[([Ss]\d+)\]/g;
    let lastIndex = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const citation = match[1].toUpperCase();
      if (sourceIds.has(citation)) {
        const button = make("button", "cite", `[${citation}]`);
        button.type = "button";
        button.dataset.citation = citation;
        button.title = `Show source ${citation}`;
        button.setAttribute("aria-label", `Show source ${citation}`);
        fragment.appendChild(button);
      } else {
        fragment.appendChild(document.createTextNode(match[0]));
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
    return fragment;
  }

  function metadataPreview(metadata) {
    if (!metadata || typeof metadata !== "object") return "";
    const hiddenKeys = new Set(["text", "chunk_text", "content", "page_content"]);
    const filtered = {};
    for (const [key, value] of Object.entries(metadata)) {
      if (hiddenKeys.has(key)) continue;
      if (value === null || value === undefined) continue;
      if (typeof value === "string" && value.length > 500) {
        filtered[key] = `${value.slice(0, 500)}…`;
      } else {
        filtered[key] = value;
      }
    }
    if (Object.keys(filtered).length === 0) return "";
    try {
      return JSON.stringify(filtered, null, 2);
    } catch {
      return "";
    }
  }

  function renderSources(sources) {
    const section = make("section", "sources");
    section.appendChild(make("h3", null, `Sources (${sources.length})`));

    for (const source of sources) {
      const details = make("details", "source");
      details.dataset.citation = String(source.citation_id || "").toUpperCase();

      const summary = make("summary");
      summary.appendChild(make("span", "cite-badge", source.citation_id || "?"));
      summary.appendChild(make("span", "vector-id", source.id || "(no id)"));
      const score = Number(source.score);
      summary.appendChild(
        make("span", "score", Number.isFinite(score) ? `score ${score.toFixed(3)}` : "")
      );
      details.appendChild(summary);

      const body = make("div", "source-body");
      body.appendChild(make("p", "source-text", source.text || "(no text)"));
      const metadata = metadataPreview(source.metadata);
      if (metadata) {
        body.appendChild(make("pre", "source-meta", metadata));
      }
      details.appendChild(body);
      section.appendChild(details);
    }
    return section;
  }

  async function copyText(text, button) {
    const original = button.textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        document.body.appendChild(helper);
        helper.select();
        document.execCommand("copy");
        helper.remove();
      }
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    window.setTimeout(() => {
      button.textContent = original;
    }, 1600);
  }

  function appendAssistantMessage(payload) {
    const wrapper = make("article", "message assistant");
    wrapper.appendChild(make("div", "avatar", "AI"));
    const body = make("div", "message-body");
    const bubble = make("div", "bubble");

    if (payload.evidence_found === false) {
      bubble.appendChild(
        make(
          "div",
          "insufficient",
          "No supporting passage was found in the index. The answer below reports that limitation instead of guessing."
        )
      );
    }

    const answer = make("div", "answer");
    const sources = Array.isArray(payload.sources) ? payload.sources : [];
    const sourceIds = new Set(
      sources.map((source) => String(source.citation_id || "").toUpperCase())
    );
    answer.appendChild(renderCitations(String(payload.answer || ""), sourceIds));
    bubble.appendChild(answer);

    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
    if (warnings.length) {
      const notes = make("details", "notes");
      notes.appendChild(make("summary", null, `Notes (${warnings.length})`));
      const list = make("ul");
      for (const warning of warnings) {
        list.appendChild(make("li", null, warning));
      }
      notes.appendChild(list);
      bubble.appendChild(notes);
    }

    if (sources.length) {
      bubble.appendChild(renderSources(sources));
    }

    const meta = make("div", "meta-row");
    meta.appendChild(
      make("span", null, payload.evidence_found === false ? "No evidence" : `${sources.length} source${sources.length === 1 ? "" : "s"}`)
    );
    if (payload.model) {
      meta.appendChild(make("span", "dot"));
      meta.appendChild(make("span", null, payload.model));
    }
    if (payload.conversation_id) {
      meta.appendChild(make("span", "dot"));
      const shortId = make("span", null, `chat ${String(payload.conversation_id).slice(0, 8)}`);
      shortId.title = `Conversation ID: ${payload.conversation_id}`;
      meta.appendChild(shortId);
    }

    const copy = make("button", "copy-button", "Copy answer");
    copy.type = "button";
    copy.addEventListener("click", () => copyText(String(payload.answer || ""), copy));
    meta.appendChild(copy);
    bubble.appendChild(meta);

    body.appendChild(bubble);
    wrapper.appendChild(body);
    el.messages.appendChild(wrapper);

    // Clicking an inline citation opens and scrolls to its source card.
    bubble.querySelectorAll("button.cite").forEach((button) => {
      button.addEventListener("click", () => {
        const target = bubble.querySelector(
          `details.source[data-citation="${button.dataset.citation}"]`
        );
        if (target) {
          target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    });

    scrollToBottom();
  }

  function showError(error, { retryable = true } = {}) {
    const titles = {
      401: "API key required",
      403: "Access denied",
      413: "Question too large",
      429: "Rate limited",
      503: "Service not ready",
      504: "Upstream timeout",
    };
    el.errorTitle.textContent = titles[error.status] || "Something went wrong";
    let message = error.message || "The request failed.";
    if (error.details && Array.isArray(error.details) && error.details.length) {
      const fields = error.details
        .map((detail) => `${detail.location}: ${detail.message}`)
        .join(" ");
      if (fields) message = `${message} ${fields}`;
    }
    el.errorMessage.textContent = message;
    el.retry.hidden = !retryable || !state.lastQuestion;
    el.errorBanner.hidden = false;

    if (error.status === 401) {
      setConnection("locked", "API key required");
      openSettings();
    }
  }

  function hideError() {
    el.errorBanner.hidden = true;
  }

  function setLoadingMessage(afterDelay) {
    el.loadingText.textContent = "Searching the index and drafting an answer…";
    window.setTimeout(() => {
      if (state.busy) {
        el.loadingText.textContent = afterDelay;
      }
    }, 2_500);
  }

  async function sendQuestion(question, { appendUser = true } = {}) {
    if (state.busy) return;
    const trimmed = question.trim();
    if (!trimmed) {
      el.input.focus();
      return;
    }

    hideError();
    if (appendUser) {
      appendUserMessage(trimmed);
    }
    state.lastQuestion = trimmed;
    const seq = ++state.requestSeq;
    setBusy(true);
    setLoadingMessage("Writing a grounded answer from the retrieved passages…");

    try {
      const payload = await api("/chat", {
        method: "POST",
        body: {
          question: trimmed,
          conversation_id: state.conversationId,
        },
      });
      if (seq !== state.requestSeq) return;
      if (payload && payload.conversation_id) {
        state.conversationId = payload.conversation_id;
      }
      appendAssistantMessage(payload || {});
      if (el.connection.dataset.state !== "ok") {
        setConnection("ok", "API online");
      }
    } catch (error) {
      if (seq !== state.requestSeq) return;
      if (error.code === "cancelled") return;
      showError(error);
      if (error.code === "network") setConnection("error", "API unreachable");
    } finally {
      if (seq === state.requestSeq) {
        setBusy(false);
        el.input.focus();
      }
    }
  }

  function submitFromInput() {
    const question = el.input.value;
    if (!question.trim() || state.busy) return;
    el.input.value = "";
    autosize();
    updateCharCount();
    sendQuestion(question, { appendUser: true });
  }

  async function resetConversation() {
    state.requestSeq += 1;
    const previousId = state.conversationId;
    state.conversationId = null;
    state.lastQuestion = null;
    setBusy(false);
    el.messages.replaceChildren();
    hideError();
    showWelcome();
    el.input.value = "";
    autosize();
    updateCharCount();
    el.input.focus();

    if (previousId) {
      try {
        await api(`/conversations/${encodeURIComponent(previousId)}`, { method: "DELETE" });
      } catch (error) {
        // History is best-effort: never block a new chat on cleanup failure.
        if (error.status === 401) setConnection("locked", "API key required");
        console.warn("Could not clear server-side conversation history:", error.message);
      }
    }
  }

  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
  }

  function updateCharCount() {
    const length = el.input.value.length;
    el.charCount.textContent = `${length} / 2000`;
    el.charCount.classList.toggle("near-limit", length > 1800);
  }

  function openSettings() {
    el.apiKeyInput.value = state.apiKey;
    if (typeof el.dialog.showModal === "function" && !el.dialog.open) {
      el.dialog.showModal();
    } else {
      el.dialog.setAttribute("open", "");
    }
    el.apiKeyInput.focus();
    el.apiKeyInput.select();
  }

  function closeSettings() {
    el.apiKeyInput.value = "";
    if (el.dialog.open) el.dialog.close();
    else el.dialog.removeAttribute("open");
  }

  function saveKey() {
    state.apiKey = el.apiKeyInput.value.trim();
    if (state.apiKey) {
      sessionStorage.setItem(STORAGE_KEY, state.apiKey);
    } else {
      sessionStorage.removeItem(STORAGE_KEY);
    }
    closeSettings();
    checkHealth();
  }

  // ---------------------------------------------------------------------
  // Event wiring
  // ---------------------------------------------------------------------
  el.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitFromInput();
  });

  el.input.addEventListener("input", () => {
    autosize();
    updateCharCount();
  });

  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitFromInput();
    }
  });

  document.querySelectorAll(".example").forEach((button) => {
    button.addEventListener("click", () => {
      const question = button.getAttribute("data-question") || button.textContent || "";
      sendQuestion(question, { appendUser: true });
    });
  });

  el.reset.addEventListener("click", () => {
    resetConversation();
  });

  el.retry.addEventListener("click", () => {
    if (!state.lastQuestion || state.busy) return;
    hideError();
    // The original question bubble is already in the transcript.
    sendQuestion(state.lastQuestion, { appendUser: false });
  });

  el.dismissError.addEventListener("click", hideError);

  el.settingsButton.addEventListener("click", openSettings);
  el.clearKey.addEventListener("click", () => {
    state.apiKey = "";
    sessionStorage.removeItem(STORAGE_KEY);
    closeSettings();
    checkHealth();
  });
  el.cancelKey.addEventListener("click", closeSettings);

  el.settingsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    saveKey();
  });

  el.dialog.addEventListener("click", (event) => {
    if (event.target === el.dialog) closeSettings();
  });

  el.dialog.addEventListener("close", () => {
    el.apiKeyInput.value = "";
  });

  // Initial state.
  autosize();
  updateCharCount();
  checkHealth();
  el.input.focus();
})();
